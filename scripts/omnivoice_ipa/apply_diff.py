"""Apply the IPA fine-tune diff (extract_diff.py) onto a BASE omnivoice_transformer.onnx,
reconstructing the fine-tuned transformer WITHOUT re-shipping the 2.45 GB merged graph.

Patches the EXTERNAL DATA FILE directly (raw bytes), so:
  - LoRA Linears: read each weight's byte range, add ΔWᵀ = ((B@A)*scale)ᵀ, write it back
    (ONNX weight is (in,out) vs PyTorch ΔW=(out,in), hence the transpose). Small per-Linear.
  - embed_tokens: seek to each CHANGED row's offset and overwrite just that row — ~22 MB of
    writes for 5,416 rows, never materializing the 620 MB table.
The MatMul nodes keep their module path (/model/llm/layers.N/self_attn/q_proj/MatMul), so we
map layer/proj -> the generically-named weight initializer they consume.

Times each phase and validates the result against the merged v5 model by logit parity.
"""
import os
import re
import shutil
import time
import numpy as np
import onnx
from onnx import numpy_helper

BASE_DIR = "/mnt/data/omnivoice_ipa/onnx_base"
BASE_ONNX = f"{BASE_DIR}/omnivoice_transformer.onnx"
BASE_DATA = f"{BASE_DIR}/omnivoice_transformer.onnx.data"
DIFF = "/mnt/data/omnivoice_ipa/onnx/ipa_diff.onnx"
OUT_ONNX = f"{BASE_DIR}/omnivoice_transformer_ipa.onnx"
OUT_DATA_NAME = "omnivoice_transformer_ipa.onnx.data"
CAP = "/mnt/data/Programming/vernacula/scripts/omnivoice_export/capture/reference.npz"
NODE_RE = re.compile(r"layers\.(\d+)/(self_attn|mlp)/(\w+_proj)/")


def ext(init):
    d = {e.key: e.value for e in init.external_data}
    return int(d["offset"]), int(d["length"])


def main():
    dm = onnx.load(DIFF)
    diff = {i.name: numpy_helper.to_array(i).astype(np.float32) if i.name != "embed_idx"
            else numpy_helper.to_array(i) for i in dm.graph.initializer}
    scale = float({p.key: p.value for p in dm.metadata_props}["lora_scale"])

    m = onnx.load(BASE_ONNX, load_external_data=False)
    inits = {i.name: i for i in m.graph.initializer}
    node2w = {}
    for nd in m.graph.node:
        if nd.op_type != "MatMul":
            continue
        mm = NODE_RE.search(nd.name)
        if mm:
            w = next((i for i in nd.input if i in inits), None)
            if w:
                node2w[(int(mm.group(1)), mm.group(2), mm.group(3))] = w

    # copy base data file -> output data file (raw fs copy; the only full-size I/O)
    out_data = f"{BASE_DIR}/{OUT_DATA_NAME}"
    t0 = time.time()
    shutil.copyfile(BASE_DATA, out_data)
    t_copy = time.time() - t0

    # patch LoRA Linears (read-modify-write each small weight range)
    t0 = time.time()
    with open(out_data, "r+b") as f:
        for (layer, sub, proj), wname in node2w.items():
            key = f"llm.layers.{layer}.{sub}.{proj}"
            A, B = diff.get(f"{key}.lora_A"), diff.get(f"{key}.lora_B")
            if A is None or B is None:
                continue
            off, ln = ext(inits[wname])
            dims = tuple(inits[wname].dims)  # (in, out)
            f.seek(off)
            W = np.frombuffer(f.read(ln), dtype=np.float32).reshape(dims).copy()
            W += ((B @ A) * scale).T.astype(np.float32)
            f.seek(off)
            f.write(W.tobytes())
        t_lin = time.time() - t0

        # patch embed rows: write ONLY the changed rows (raw bytes), no full-table copy
        t0 = time.time()
        emb = inits["model.llm.embed_tokens.weight"]
        eoff, _ = ext(emb)
        hidden = int(emb.dims[1])
        rowbytes = hidden * 4
        rows = diff["embed_rows"].astype(np.float32)
        idx = diff["embed_idx"].astype(np.int64)
        for r, row in zip(idx, rows):
            f.seek(eoff + int(r) * rowbytes)
            f.write(row.tobytes())
        t_emb = time.time() - t0

    # point the graph at the new data file + save (proto only, tiny)
    for i in m.graph.initializer:
        for e in i.external_data:
            if e.key == "location":
                e.value = OUT_DATA_NAME
    t0 = time.time()
    onnx.save_model(m, OUT_ONNX, save_as_external_data=False)  # data already on disk; write graph
    t_save = time.time() - t0

    print(f"copy base .onnx.data (2.45 GB fs copy): {t_copy:.1f} s")
    print(f"patch {len(node2w)} LoRA Linears        : {t_lin:.2f} s")
    print(f"patch {len(idx)} embed rows (raw bytes) : {t_emb:.2f} s  <- was ~2 s via full-table round-trip")
    print(f"write graph proto                     : {t_save:.2f} s")
    print(f"=> total                              : {t_copy + t_lin + t_emb + t_save:.1f} s")

    # --- parity vs merged v4 PyTorch ---
    import sys, torch, onnxruntime as ort
    sys.path.insert(0, "/mnt/data/Programming/vernacula/scripts/omnivoice_export")
    from omnivoice.models.omnivoice import OmniVoice
    from export_omnivoice import TransformerWrapper
    from peft import PeftModel
    cap = np.load(CAP)
    a = ort.InferenceSession(OUT_ONNX, providers=["CPUExecutionProvider"]).run(
        ["logits"], {"input_ids": cap["tf_input_ids"].astype(np.int64),
                     "audio_mask": cap["tf_audio_mask"].astype(bool),
                     "attention_mask": cap["tf_attention_mask"].astype(bool)})[0]
    mdl = OmniVoice.from_pretrained("/mnt/data/models/omnivoice/k2-fsa-OmniVoice",
                                    device_map="cpu", dtype=torch.float32)
    mdl = PeftModel.from_pretrained(
        mdl, "/mnt/data/omnivoice_ipa/train/checkpoints_v5/checkpoint-4000").merge_and_unload().eval()
    try:
        mdl.llm.config._attn_implementation = "sdpa"
    except Exception:
        pass
    with torch.no_grad():
        b = TransformerWrapper(mdl)(torch.from_numpy(cap["tf_input_ids"]),
                                    torch.from_numpy(cap["tf_audio_mask"]),
                                    torch.from_numpy(cap["tf_attention_mask"])).numpy()
    agree = float(np.mean(a.argmax(-1) == b.argmax(-1)))
    print(f"\nparity folded-vs-merged-v5: argmax {agree*100:.3f}%  max|Δlogit| {np.abs(a-b).max():.2e}")
    print("PASS" if agree > 0.999 else "CHECK")


if __name__ == "__main__":
    main()
