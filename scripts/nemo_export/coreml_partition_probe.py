#!/usr/bin/env python3
"""Measure what an execution provider actually does with a Sortformer graph:
partition count, load time, inference time, and parity against a reference model.

The partition count is the number that matters for CoreML (see
docs/coreml_onnx_playbook.md) and ONNX Runtime only reports it as a log line from
inside the C++ EP, so this script captures the EP's stderr rather than trying to
read it off the session.

Typical use -- the go/no-go for issue #162, on Apple Silicon:

    python scripts/nemo_export/coreml_partition_probe.py \\
        --model diar_streaming_sortformer_4spk-v2.1.coreml.onnx \\
        --ep coreml --cache-dir ~/.cache/vernacula-coreml \\
        --reference diar_streaming_sortformer_4spk-v2.1.onnx

Run it once per EP (`--ep cpu`, `--ep webgpu`, `--ep coreml`) and once per ORT
version to compare. Nothing here is macOS-specific except which EPs exist.

Inputs are synthesized from the model's own signature: float buffers get
seeded random-normal values, and a `<buffer>_lengths` input gets that buffer's
own frame count -- the steady state both graphs are specialized for. A short
final chunk is deliberately NOT exercised here; the steady-state graph is not
valid for one, which is the whole point of the contract.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import re
import statistics
import sys
import tempfile
import time

import numpy as np

LEVELS = {
    "disable": "ORT_DISABLE_ALL",
    "basic": "ORT_ENABLE_BASIC",
    "extended": "ORT_ENABLE_EXTENDED",
    "all": "ORT_ENABLE_ALL",
}

# CoreMLExecutionProvider::GetCapability logs this at warning level.
PARTITION_RE = re.compile(
    r"number of partitions supported by \w+:\s*(\d+)"
    r".*?number of nodes in the graph:\s*(\d+)"
    r".*?number of nodes supported by \w+:\s*(\d+)",
    re.S,
)


@contextlib.contextmanager
def capture_native_stderr():
    """Redirect fd 2 to a temp file so the EP's C++ logging is readable.

    Reassigning sys.stderr does not work -- the log line is written by the native
    library, which holds the file descriptor directly.
    """
    with tempfile.TemporaryFile(mode="w+") as tmp:
        saved = os.dup(2)
        sys.stderr.flush()
        os.dup2(tmp.fileno(), 2)
        try:
            yield tmp
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
            tmp.seek(0)


def build_options(ort, level: str, verbose: bool):
    so = ort.SessionOptions()
    so.graph_optimization_level = getattr(ort.GraphOptimizationLevel, LEVELS[level])
    # Partition counts are logged at WARNING; node placement needs VERBOSE.
    so.log_severity_level = 0 if verbose else 2
    if verbose:
        so.log_verbosity_level = 1
    return so


def providers_for(ep: str, cache_dir: str | None):
    if ep == "cpu":
        return ["CPUExecutionProvider"]
    if ep == "webgpu":
        return ["WebGpuExecutionProvider", "CPUExecutionProvider"]
    opts = {
        # ML Program is CoreML's current IR; the legacy NeuralNetwork format is frozen.
        "ModelFormat": "MLProgram",
        "MLComputeUnits": "ALL",
    }
    if cache_dir:
        opts["ModelCacheDirectory"] = os.path.expanduser(cache_dir)
    return [("CoreMLExecutionProvider", opts), "CPUExecutionProvider"]


def make_session(ort, model: str, ep: str, level: str, cache_dir: str | None, verbose: bool):
    """Returns (session, seconds, captured EP log)."""
    so = build_options(ort, level, verbose)
    with capture_native_stderr() as log:
        t0 = time.perf_counter()
        sess = ort.InferenceSession(model, so, providers=providers_for(ep, cache_dir))
        elapsed = time.perf_counter() - t0
        text = log.read()
    return sess, elapsed, text


def synth_feed(sess, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    shapes = {i.name: i.shape for i in sess.get_inputs()}
    feed = {}
    for i in sess.get_inputs():
        if i.type == "tensor(int64)":
            if not i.name.endswith("_lengths"):
                raise SystemExit(f"cannot synthesize integer input {i.name}")
            buf = i.name[: -len("_lengths")]
            if buf not in shapes:
                raise SystemExit(f"cannot infer a length for {i.name}: no input named {buf}")
            feed[i.name] = np.full([1], shapes[buf][1], np.int64)
        else:
            if any(not isinstance(d, int) for d in i.shape):
                raise SystemExit(
                    f"--model must be static; input {i.name} has shape {i.shape}. "
                    "Point --model at the CoreML variant and the dynamic graph at "
                    "--reference, not the other way round."
                )
            dtype = np.float16 if i.type == "tensor(float16)" else np.float32
            feed[i.name] = rng.standard_normal(i.shape).astype(dtype)
    return feed


def report_placement(log: str) -> None:
    m = PARTITION_RE.search(log)
    if m:
        parts, total, supported = (int(x) for x in m.groups())
        note = "  <- single partition" if parts == 1 else "  <- every extra partition is a copy + sync"
        print(f"  partitions      {parts} ({supported}/{total} nodes on the EP){note}")
    else:
        print("  partitions      not reported (the CPU and WebGPU EPs do not log this)")
    fell_back = re.findall(r"Node\(s\) placed on \[CPUExecutionProvider\][^\n]*", log)
    for line in fell_back[:1]:
        print(f"  {line.strip()}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Static Sortformer graph to probe")
    ap.add_argument("--ep", default="coreml", choices=["coreml", "cpu", "webgpu"])
    ap.add_argument("--opt-level", default="basic", choices=sorted(LEVELS),
                    help="Session graph optimization level (default basic -- the only "
                         "level a post-processed graph reliably loads at)")
    ap.add_argument("--cache-dir", help="CoreML ModelCacheDirectory; enables the warm-load "
                                       "measurement by loading a second time")
    ap.add_argument("--reference", help="Dynamic graph to check outputs against, on the CPU EP")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--verbose", action="store_true",
                    help="VERBOSE ORT logging, so per-node EP placement is listed")
    args = ap.parse_args()

    import onnxruntime as ort

    model = os.path.expanduser(args.model)
    print(f"onnxruntime     {ort.__version__}")
    print(f"available EPs   {', '.join(ort.get_available_providers())}")
    print(f"model           {model} ({os.path.getsize(model) / 1e6:.0f} MB)")
    print(f"ep / opt level  {args.ep} / {LEVELS[args.opt_level]}")

    try:
        sess, cold, log = make_session(ort, model, args.ep, args.opt_level, args.cache_dir, args.verbose)
    except Exception as exc:
        # A load failure IS a result -- ORT_ENABLE_EXTENDED and above are expected to
        # throw on a post-processed graph, and CoreML rejects a graph with unbounded dims.
        print(f"\nLOAD FAILED at {LEVELS[args.opt_level]}: {str(exc).strip().splitlines()[-1]}")
        sys.exit(1)

    print(f"\nload")
    print(f"  cold            {cold:.2f} s")
    if args.cache_dir:
        _, warm, _ = make_session(ort, model, args.ep, args.opt_level, args.cache_dir, False)
        print(f"  warm            {warm:.2f} s")
    actually_used = sess.get_providers()
    print(f"  providers used  {', '.join(actually_used)}")
    if args.ep != "cpu" and actually_used[0] == "CPUExecutionProvider":
        print("  ⚠ the requested EP did not take the graph -- numbers below are CPU numbers")
    report_placement(log)

    feed = synth_feed(sess, args.seed)
    out_names = [o.name for o in sess.get_outputs()]
    for _ in range(args.warmup):
        sess.run(None, feed)
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        got = sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1e3)
    print(f"\ninference over {args.runs} runs")
    print(f"  median          {statistics.median(times):.1f} ms")
    print(f"  min / max       {min(times):.1f} / {max(times):.1f} ms")

    if args.reference:
        ref_sess = ort.InferenceSession(
            os.path.expanduser(args.reference), build_options(ort, "basic", False),
            providers=["CPUExecutionProvider"])
        ref_feed = {}
        for i in ref_sess.get_inputs():
            if i.name in feed:
                ref_feed[i.name] = feed[i.name]
            else:
                # The reference declares inputs the specialized graph baked away.
                ref_feed.update(synth_feed_missing(i, feed))
        ref = dict(zip([o.name for o in ref_sess.get_outputs()], ref_sess.run(None, ref_feed)))
        mine = dict(zip(out_names, got))
        print(f"\nparity vs {os.path.basename(args.reference)} (CPU EP, steady state)")
        for k in ("spkcache_fifo_chunk_preds", "chunk_pre_encode_embs"):
            if k not in ref or k not in mine:
                continue
            a, b = ref[k].astype(np.float32), mine[k].astype(np.float32)
            if a.shape != b.shape:
                print(f"  {k:26} SHAPE {a.shape} vs {b.shape}")
                continue
            d = np.abs(a - b)
            print(f"  {k:26} maxAbs={d.max():.3E}  rms={np.sqrt((d ** 2).mean()):.3E}")


def synth_feed_missing(inp, feed: dict) -> dict:
    """A `<buffer>_lengths` the specialized graph no longer declares. Its steady-state
    value is the buffer's own frame count, which is what was baked in."""
    if not inp.name.endswith("_lengths"):
        raise SystemExit(f"reference wants {inp.name}, which --model does not provide")
    buf = inp.name[: -len("_lengths")]
    if buf not in feed:
        raise SystemExit(f"reference wants {inp.name} but --model has no {buf} input")
    return {inp.name: np.full([1], feed[buf].shape[1], np.int64)}


if __name__ == "__main__":
    main()
