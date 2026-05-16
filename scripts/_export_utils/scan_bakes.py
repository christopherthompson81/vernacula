"""Find baked-in trace-time shape constants in an ONNX graph.

When `torch.jit.trace` records a sequence-length-dependent op (Reshape,
Expand, Where, Concat-of-shape), it sometimes collapses a SymInt to a
Python literal — e.g., a `view(80, T)` where T came from `x.size(-1)`
becomes a `Reshape` with shape `[80, 1024]` baked as a constant. The
exported ONNX only works at the exact trace-time length unless every
such collapse is eliminated.

This module walks every `Constant` node, decodes small int tensors, and
reports the consumer ops whose constant inputs contain "suspect" values
(integers derived from the trace-time sequence length: T, 2*T, T/2, etc.).
Hits are grouped by ONNX op scope (e.g. `/decoder/Reshape_9`), which maps
back to the PyTorch module path that produced the trace.

Why this matters:
- The "patch one bake, re-export, discover the next" loop is slow and
  misses interactions. An up-front inventory of every bake site in a
  single export pass scopes the fix work in one shot.
- A passing run (zero hits for legitimate suspect values) is a strong
  signal that the export is genuinely dynamic-shape.

Caveats:
- Suspect values must include legitimate model constants (e.g., `inner_dim`,
  `num_heads`, `head_dim`) only when they coincidentally equal a trace
  dimension — those are false positives. Compare against a second export
  at a different trace length to disambiguate.
- Only flags constants used directly by a consumer; doesn't currently
  follow Concat/Gather chains. Those can hide trace-derived ints inside
  composed shape tensors. Good enough in practice.

CLI:
    python scripts/_export_utils/scan_bakes.py <onnx_path> --suspect 512 1024 [...]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import onnx
from onnx import numpy_helper


def _decode_constants(graph: onnx.GraphProto) -> dict[str, list[int]]:
    """Return {tensor_name: int_list} for every small-int Constant node."""
    out: dict[str, list[int]] = {}
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                try:
                    arr = numpy_helper.to_array(attr.t)
                except Exception:
                    continue
                if arr.dtype.kind in "iu" and arr.size <= 32:
                    out[node.output[0]] = arr.flatten().tolist()
            elif attr.name == "value_ints":
                out[node.output[0]] = list(attr.ints)
            elif attr.name == "value_int":
                out[node.output[0]] = [attr.i]
    return out


def _top_scope(name: str) -> str:
    parts = name.lstrip("/").split("/")
    return "/" + "/".join(parts[:2]) if len(parts) >= 2 else "/" + name.lstrip("/")


def scan(
    onnx_path: Path,
    suspect_vals: Iterable[int],
) -> list[tuple[str, str, list[int], str, list[int]]]:
    """Scan an ONNX file for nodes whose constant inputs contain suspect ints.

    Returns a list of `(node_name, op_type, hits, input_name, full_const)`
    tuples. The caller can format/group as needed; `print_report` does
    a reasonable default.
    """
    suspect = set(suspect_vals)
    model = onnx.load(str(onnx_path), load_external_data=False)
    consts = _decode_constants(model.graph)
    hits = []
    for node in model.graph.node:
        for inp in node.input:
            if inp not in consts:
                continue
            vals = consts[inp]
            seen = [v for v in vals if v in suspect]
            if seen:
                hits.append((node.name, node.op_type, seen, inp, vals))
    return hits


def print_report(
    onnx_path: Path,
    hits: list[tuple[str, str, list[int], str, list[int]]],
    samples_per_scope: int = 5,
) -> None:
    by_scope: dict[str, list] = defaultdict(list)
    for h in hits:
        by_scope[_top_scope(h[0])].append(h)
    print(f"Scanned: {onnx_path}")
    print(f"  {len(hits)} consumer nodes use baked suspect constants")
    print(f"  across {len(by_scope)} top-level scope(s)")
    op_counter: Counter = Counter()
    for scope, scope_hits in sorted(by_scope.items(), key=lambda kv: -len(kv[1])):
        print(f"\n--- {scope}  ({len(scope_hits)} hits) ---")
        for name, op, seen, _inp, vals in scope_hits[:samples_per_scope]:
            op_counter[op] += 1
            print(f"  {name}  [{op}]  seen={seen}  shape={vals}")
        for name, op, _seen, _inp, _vals in scope_hits[samples_per_scope:]:
            op_counter[op] += 1
        if len(scope_hits) > samples_per_scope:
            print(f"  ... +{len(scope_hits) - samples_per_scope} more")
    if op_counter:
        print("\nOp-type tally across all hits:")
        for op, n in op_counter.most_common():
            print(f"  {op}: {n}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("onnx_path", type=Path, help="Path to the .onnx file to scan")
    p.add_argument(
        "--suspect", type=int, nargs="+", required=True,
        help="Trace-derived integers that signal a baked dim (e.g. 512 1024 "
             "for a 512-token trace with 2x mel upsample). Compare against a "
             "second export at a different trace length to rule out coincidental "
             "matches with legitimate model constants.",
    )
    p.add_argument(
        "--samples-per-scope", type=int, default=5,
        help="How many hits to print per scope before summarizing 'and more'.",
    )
    args = p.parse_args()
    hits = scan(args.onnx_path, args.suspect)
    print_report(args.onnx_path, hits, samples_per_scope=args.samples_per_scope)


if __name__ == "__main__":
    main()
