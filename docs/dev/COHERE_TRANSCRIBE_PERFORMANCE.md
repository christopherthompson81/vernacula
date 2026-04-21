Language detection seems bad. We're going to have to implement forcing language definition.

# Step 1 — get segment JSON (C# CLI, no transcription)
dotnet run --project ./public/src/Vernacula.CLI/Vernacula.CLI.csproj -- \
    --audio data/test_audio/en-US/en-US_sample_01.wav \
    --model ~/.local/share/Vernacula/models \
    --diarization vad --skip-asr --export-format json \
    --output /tmp/segs.json

# Step 2 — Python language diagnostic
python scripts/cohere_export/test_language_detection.py \
    --model-dir ~/.local/share/Vernacula/models/cohere_transcribe \
    --audio data/test_audio/en-US/en-US_sample_01.wav \
    --segments /tmp/segs.json \
    --max-segments 50

---

dotnet run --project ./public/src/Vernacula.CLI/Vernacula.CLI.csproj -- --audio data/test_audio/en-US/en-US_sample_01.wav --model ~/.local/share/Vernacula/models --asr cohere --diarization vad --language en --benchmark

---

Here's what I see as the main levers, roughly in order of impact-to-effort ratio:

Decoder (90% of time — most leverage here)
1. ORT IO binding
The session log warns "3 Memcpy nodes added for CUDAExecutionProvider" — ORT is copying certain inputs/outputs between CPU and GPU on every step. IOBinding (session.RunWithBinding) pins named tensors to a CUDA device buffer and avoids those round-trips. This is a pure C# change, no re-export needed.

2. FP16 decoder weights
The decoder is memory-bandwidth-limited (reading ~6 GB of weights per step). FP16 halves the HBM reads. ORT has a built-in graph optimization to convert float32 models to float16 (ConvertModelToFloat16 Python utility, or --convert_float_to_float16 via onnxmltools). No architectural change — just re-export/convert. Could give ~1.5× decoder throughput.

3. Context-block prefill
The decoder currently emits ~8 context-block tokens (BOS → startofcontext → startoftranscript → emotion → language × 2 → pnc → itn → nodiarize) one token at a time, each requiring a full decoder forward pass. When the language is forced, the entire context block is deterministic. We could modify decoder_init.onnx to accept a multi-token prefix [B, T_prefix] instead of just BOS, collapsing 8 serial steps into one — a ~25% reduction in total decode steps for typical segments.

4. Encoder-decoder pipeline
While the decoder is running on batch N, the encoder could be computing hidden states for batch N+1 on a second CUDA stream. This hides ~6 s of encoder time behind decoder compute. ORT supports per-session CUDA stream assignment via SessionOptions.SetCudaSessionStreamBatch. Pure C# change; would require a producer-consumer queue between encoder and decoder stages.

Encoder (9% of time — smaller but non-trivial)
5. Reduce encoder padding waste
Currently we pad all segments in a batch to F_max (the longest segment's mel frame count). Since we sort by duration, segments in a batch are already similar lengths — but we could tighten the batch-splitting criterion to also bound the ratio F_max / F_min, limiting how much padding waste occurs.

6. Batch mel on GPU
mel.onnx was exported with B=1 only and runs on CPU. Re-exporting with a dynamic batch axis would let all segments in a batch run mel together on GPU, though this is only ~250 ms total (< 1% of time).

System / architectural
7. TensorRT EP for decoder_step
The decoder step is the hottest ONNX execution path — fixed-shape per-step (only tPast grows) and structurally repetitive. TRT fuses ops and selects optimal CUDA kernels at compile time. Compiling decoder_step.onnx into a TRT engine with FP16 could give 2–3× over ORT CUDA. Highest effort but highest ceiling.

8. INT8 / GPTQ quantization of decoder
4-bit or 8-bit quantized decoder weights would reduce HBM reads to 1/8 or 1/4 of FP32. Quality loss is recoverable with calibration. Would require onnxruntime-gpu quantization tools.

Low-hanging fruit to try first
IOBinding (#1) and FP16 conversion (#2) are the lowest-effort options with meaningful expected gains. Context-block prefill (#3) is medium effort with a guaranteed ~25% step-count reduction when --language is forced. The encoder-decoder pipeline (#4) is a clean architectural win that hides latency that's currently pure dead time.

---

.venv-cohere-asr/bin/python3 scripts/cohere_export/export_cohere_decoder_kv.py \
    --model-dir ~/.local/share/Vernacula/models/cohere_transcribe \
    --overwrite \
    --device cuda \
    --skip-init 2>&1

.venv-cohere-asr/bin/python3 scripts/cohere_export/export_cohere_decoder_kv.py \
    --model-dir ~/.local/share/Vernacula/models/cohere_transcribe \
    --overwrite \
    --device cuda