# DeepFilterNet3 Streaming Export

Exports DeepFilterNet3 as three split ONNX models with explicit GRU hidden-state I/O, enabling chunk-by-chunk inference in C#.

The standard single-call export processes all T frames at once, which gives poor GPU utilisation because GRU layers must step sequentially. The streaming export processes audio in chunks of ~100 frames (~1 s), carrying GRU state across chunks.

## Files

- `export_df3_streaming.py` — exports the three streaming ONNX models and writes `streaming_meta.json`
- `requirements.txt` — export dependencies

## Environment

```bash
pip install -r scripts/deepfilternet3_export/requirements.txt
```

## Export

```bash
python scripts/deepfilternet3_export/export_df3_streaming.py \
  --out external/deepfilternet_onnx_streaming \
  --opset 14
```

Outputs:

- `enc.onnx`
- `erb_dec.onnx`
- `df_dec.onnx`
- `streaming_meta.json` — hidden-state shapes and chunk size

## ONNX Contract

**enc.onnx**

| Name | Shape | Description |
|---|---|---|
| `feat_erb` | `[1, 1, T, 32]` | ERB filterbank features |
| `feat_spec` | `[1, 2, T, 96]` | Complex spectrogram features |
| `h_enc` | `[1, 1, 256]` | Encoder GRU hidden state in |
| `e0..e3` | `[1, 64, T, *]` | Skip connections |
| `emb` | `[1, T, 512]` | Embedding |
| `c0` | `[1, 64, T, 96]` | DF conv features |
| `lsnr` | `[1, T, 1]` | Log SNR estimate |
| `h_enc_out` | `[1, 1, 256]` | Encoder GRU hidden state out |

**erb_dec.onnx**

| Name | Shape | Description |
|---|---|---|
| `emb` | `[1, T, 512]` | Embedding (from enc) |
| `e3..e0` | `[1, 64, T, *]` | Skip connections (from enc) |
| `h_erb` | `[2, 1, 256]` | ERB decoder GRU hidden state in |
| `m` | `[1, 1, T, 32]` | ERB mask |
| `h_erb_out` | `[2, 1, 256]` | ERB decoder GRU hidden state out |

**df_dec.onnx**

| Name | Shape | Description |
|---|---|---|
| `emb` | `[1, T, 512]` | Embedding (from enc) |
| `c0` | `[1, 64, T, 96]` | DF conv features (from enc) |
| `h_df` | `[2, 1, 256]` | DF decoder GRU hidden state in |
| `coefs` | `[1, T, 96, 10]` | Deep filtering coefficients |
| `h_df_out` | `[2, 1, 256]` | DF decoder GRU hidden state out |

`T` is a dynamic axis — use any chunk size consistent with `streaming_meta.json`.
