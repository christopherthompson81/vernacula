"""Extract a SMALL distributable diff from the IPA fine-tune, to be applied over the base
OmniVoice transformer (see apply_diff.py).

The fine-tune = LoRA on q/k/v/o/gate/up/down (true low-rank) + a fully-retrained embed_tokens.
But embed_tokens is effectively SPARSE: only the IPA-relevant rows learned; the rest just drifted
by weight decay (~0.0003). We keep every row whose max|Δ| exceeds a threshold well above that
floor (default 0.001) — inclusive, so it captures both boosted IPA tokens AND *suppressed*
orthographic tokens (a real, learned change), per design discussion.

Output (fp16 safetensors): LoRA A/B factors per module + the changed embed rows (values) + their
indices + metadata (scale, threshold). ~30 MB vs the 2.45 GB merged transformer (~80x smaller).
"""
import glob
import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper
from safetensors import safe_open

BASE = "/mnt/data/models/omnivoice/k2-fsa-OmniVoice"
ADAPTER = "/mnt/data/omnivoice_ipa/train/checkpoints_v5/checkpoint-4000/adapter_model.safetensors"
OUT = "/mnt/data/omnivoice_ipa/onnx/ipa_diff.onnx"
EMBED_THRESHOLD = 0.001   # rows with max|Δ| below this are weight-decay drift → keep base
LORA_ALPHA, LORA_R = 32, 16   # scale = alpha/r = 2.0


def base_embed():
    for f in glob.glob(f"{BASE}/*.safetensors"):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                if "embed_tokens.weight" in k:
                    return h.get_tensor(k).float()
    raise RuntimeError("base embed_tokens not found")


def main():
    inits = []
    with safe_open(ADAPTER, "pt") as h:
        keys = list(h.keys())
        # LoRA A/B — fp16 initializers, named by the peft module path (layer.N....proj_lora_A)
        # so the fold maps them to the base graph's Linear weights.
        for k in keys:
            if ".lora_A.weight" in k or ".lora_B.weight" in k:
                name = k.replace("base_model.model.", "").replace(".weight", "")
                inits.append(numpy_helper.from_array(
                    h.get_tensor(k).half().numpy(), name))
        ek = [k for k in keys if "embed_tokens" in k][0]
        ft = h.get_tensor(ek).float()
    d = (ft - base_embed()).abs().amax(dim=1)
    idx = torch.nonzero(d > EMBED_THRESHOLD, as_tuple=True)[0]
    inits.append(numpy_helper.from_array(ft[idx].half().numpy(), "embed_rows"))
    inits.append(numpy_helper.from_array(idx.to(torch.int32).numpy(), "embed_idx"))

    # a ModelProto that carries the diff as initializers (empty graph — it's a tensor container,
    # not a runnable model). C# reads graph.initializer via the same ONNX protobuf reader it uses
    # for the base transformer; metadata carries the LoRA scale.
    graph = helper.make_graph([], "ipa_diff", [], [], initializer=inits)
    m = helper.make_model(graph, producer_name="omnivoice_ipa_extract_diff")
    m.metadata_props.append(onnx.StringStringEntryProto(key="lora_scale", value=str(LORA_ALPHA / LORA_R)))
    m.metadata_props.append(onnx.StringStringEntryProto(key="embed_threshold", value=str(EMBED_THRESHOLD)))
    onnx.save_model(m, OUT)

    import os
    n_lora = sum(1 for i in inits if "lora_" in i.name)
    print(f"wrote {OUT}: {os.path.getsize(OUT)/1e6:.1f} MB")
    print(f"  LoRA initializers: {n_lora} (fp16), embed rows: {len(idx)}/{ft.shape[0]} (>{EMBED_THRESHOLD})")


if __name__ == "__main__":
    main()
