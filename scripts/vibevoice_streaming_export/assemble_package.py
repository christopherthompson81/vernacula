#!/usr/bin/env python
"""Assemble a publishable VibeVoice-ASR-Streaming package: the fp16 audio encoder, the
INT8 GQA decoder, and the metadata the C# backend reads.

Both graphs are re-saved with external data so every package has the same file set
(`*.onnx` + `*.onnx.data`) regardless of whether a graph happened to fit in one protobuf.
The downloader's asset list is static, so a package that omits a file it expects would 404
on install.
"""
import argparse, json, shutil
from pathlib import Path

import onnx

METADATA = ["config.json", "preprocessor_config.json", "tokenizer_config.json",
            "tokenizer.json", "export-report.json"]


def save_external(src: Path, dst: Path):
    m = onnx.load(str(src))
    onnx.save(m, str(dst), save_as_external_data=True, all_tensors_to_one_file=True,
              location=dst.name + ".data", size_threshold=1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-dir", required=True, help="package holding the fp16 audio_encoder.onnx")
    ap.add_argument("--decoder-dir", required=True, help="package holding the INT8 decoder_gqa.onnx")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    enc, dec, out = Path(args.encoder_dir), Path(args.decoder_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*"):
        f.unlink()

    save_external(enc / "audio_encoder.onnx", out / "audio_encoder.onnx")
    save_external(dec / "decoder_gqa.onnx", out / "decoder_gqa.onnx")
    for name in METADATA:
        src = dec / name
        if not src.exists():
            src = enc / name
        shutil.copyfile(src.resolve(), out / name)

    rep = json.loads((out / "export-report.json").read_text())
    rep["package"] = {"audio_encoder": "float16", "decoder": "GroupQueryAttention + INT8 weights"}
    (out / "export-report.json").write_text(json.dumps(rep, indent=1))
    total = sum(f.stat().st_size for f in out.iterdir())
    print(f"{out}: {len(list(out.iterdir()))} files, {total / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
