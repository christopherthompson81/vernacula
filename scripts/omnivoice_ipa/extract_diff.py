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
import json
import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "/mnt/data/models/omnivoice/k2-fsa-OmniVoice"
ADAPTER = "/mnt/data/omnivoice_ipa/train/checkpoints_v4/checkpoint-4000/adapter_model.safetensors"
OUT = "/mnt/data/omnivoice_ipa/onnx/ipa_diff.safetensors"
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
    tensors = {}
    meta = {"scale": str(LORA_ALPHA / LORA_R), "embed_threshold": str(EMBED_THRESHOLD)}
    with safe_open(ADAPTER, "pt") as h:
        keys = list(h.keys())
        # LoRA A/B — store fp16, keyed by the peft module path (layer.N....proj)
        for k in keys:
            if ".lora_A.weight" in k or ".lora_B.weight" in k:
                # e.g. base_model.model.llm.layers.0.self_attn.q_proj.lora_A.weight
                tensors[k.replace("base_model.model.", "")] = h.get_tensor(k).half()
        # sparse embed rows
        ek = [k for k in keys if "embed_tokens" in k][0]
        ft = h.get_tensor(ek).float()
    d = (ft - base_embed()).abs().amax(dim=1)
    idx = torch.nonzero(d > EMBED_THRESHOLD, as_tuple=True)[0]
    tensors["embed_rows"] = ft[idx].half()
    tensors["embed_idx"] = idx.to(torch.int32)

    n_lora = len([k for k in tensors if "lora_" in k])
    meta["n_lora_tensors"] = str(n_lora)
    meta["n_embed_rows"] = str(len(idx))
    save_file(tensors, OUT, metadata=meta)

    import os
    sz = os.path.getsize(OUT) / 1e6
    print(f"wrote {OUT}: {sz:.1f} MB")
    print(f"  LoRA tensors: {n_lora} (fp16), embed rows: {len(idx)}/{ft.shape[0]} (>{EMBED_THRESHOLD})")


if __name__ == "__main__":
    main()
