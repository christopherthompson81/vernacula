#!/usr/bin/env python
"""Weight-only INT4/INT8 quantization of the streaming decoder's MatMul weights
(ORT MatMulNBits). The KV cache, attention maths and the conv audio encoder stay float;
only the Qwen2 linear layers change. Writes a new package directory that shares the
audio encoder and metadata with the source package.

Parity is measured the same way as everything else: WER against the deterministic torch
reference, speaker count unchanged.
"""
import argparse, collections, json, shutil, time
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer


def fold_weight_transposes(model) -> int:
    """Rewrite MatMul(x, Transpose(W)) as MatMul(x, W^T) with W^T an initializer.

    torch.onnx keeps nn.Linear weights as [out, in] initializers behind a Transpose, so the
    MatMul's B input is a node output and MatMulNBitsQuantizer skips it ("doesn't have const
    weight"). ORT's constant folding leaves these alone (the tensors live in external data),
    so fold them here.
    """
    import numpy as np
    from onnx import numpy_helper

    inits = {i.name: i for i in model.graph.initializer}
    prod = {o: n for n in model.graph.node for o in n.output}
    consumers = collections.Counter(i for n in model.graph.node for i in n.input)
    folded, dead = 0, []
    for node in model.graph.node:
        if node.op_type != "MatMul" or len(node.input) < 2:
            continue
        t = prod.get(node.input[1])
        if t is None or t.op_type != "Transpose" or t.input[0] not in inits:
            continue
        perm = next((list(a.ints) for a in t.attribute if a.name == "perm"), None)
        w = inits[t.input[0]]
        if perm != [1, 0] or len(w.dims) != 2:
            continue
        name = w.name + "__T"
        if name not in inits:
            arr = numpy_helper.to_array(w)
            new = numpy_helper.from_array(np.ascontiguousarray(arr.T), name)
            model.graph.initializer.append(new)
            inits[name] = new
        node.input[1] = name
        folded += 1
        if consumers[t.output[0]] == 1:
            dead.append(t)
    for t in dead:
        model.graph.node.remove(t)
    return folded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="exported package directory")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8])
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--decoder", default="decoder_single.onnx")
    ap.add_argument("--keep-lm-head", action="store_true",
                    help="leave the vocab projection in its original precision")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name.startswith("decoder_") or f.name == "export-report.json":
            continue
        if not (dst / f.name).exists():
            (dst / f.name).symlink_to(f.resolve())   # audio encoder + tokenizer files are unchanged

    model = onnx.load(str(src / args.decoder))
    folded = fold_weight_transposes(model)
    print(f"folded {folded} Transpose(initializer) weights into initializers")
    exclude = []
    if args.keep_lm_head:
        # The vocab projection is the single largest MatMul and feeds the argmax directly.
        exclude = [n.name for n in model.graph.node
                   if n.op_type in ("MatMul", "Gemm") and any("lm_head" in i or "embed_tokens" in i for i in n.input)]
    t0 = time.time()
    q = MatMulNBitsQuantizer(model, block_size=args.block_size, is_symmetric=True,
                             bits=args.bits, nodes_to_exclude=exclude)
    q.process()
    onnx.save(q.model.model, str(dst / args.decoder), save_as_external_data=True,
              all_tensors_to_one_file=True, location=args.decoder + ".data", size_threshold=1024)
    took = time.time() - t0

    rep = json.loads((src / "export-report.json").read_text())
    rep["quantization"] = {"bits": args.bits, "block_size": args.block_size, "symmetric": True,
                           "excluded_nodes": len(exclude), "source": str(src), "seconds": round(took, 1)}
    (dst / "export-report.json").write_text(json.dumps(rep, indent=1))
    print(f"wrote {dst/args.decoder} in {took:.0f}s ({len(exclude)} nodes excluded)")


if __name__ == "__main__":
    main()
