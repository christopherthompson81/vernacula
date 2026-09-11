using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Models;

namespace Vernacula.Tts.Base;

/// <summary>
/// One Kokoro forward pass: 24 kHz audio, the per-input-token predicted durations
/// (<paramref name="PredDur"/>, duration units), and the encoded token ids
/// (<paramref name="InputIds"/>, including the bracketing pad tokens). PredDur and
/// InputIds are index-aligned and together let callers recover word timings.
/// </summary>
public sealed record KokoroOutput(float[] Audio, long[] PredDur, long[] InputIds);

/// <summary>
/// hexgrad/Kokoro-82M TTS inference over the Vernacula-exported
/// <c>kokoro.onnx</c> graph (StyleTTS2 / iSTFTNet, 24 kHz mono).
///
/// The graph entry point is <c>forward_with_tokens</c> — the G2P frontend is
/// outside the graph. Callers supply a Kokoro-alphabet phoneme string (from
/// <see cref="KokoroFormat.Render"/>); this class handles
/// tokenization (<see cref="KokoroVocab"/>), voice-style selection, and the
/// ONNX run. See scripts/kokoro_export and docs/kokoro_onnx_investigation.md.
///
/// Not thread-safe — matches the underlying ORT <see cref="InferenceSession"/>.
/// Construct one instance per concurrent caller.
/// </summary>
public sealed class Kokoro : IDisposable
{
    /// <summary>Output sample rate of the Kokoro decoder.</summary>
    public const int SampleRate = 24_000;

    private const int StyleDim = 256;   // ref_s width
    private const int VoiceRows = 510;  // voice-pack index range (= max phoneme-string length)
    private const int SamplesPerFrame = 600;  // model constant: one duration unit at 24 kHz

    private readonly InferenceSession _session;
    private readonly string _voicesDir;
    // kokoro_batched.onnx takes an extra input_lengths and a leading batch axis; kokoro.onnx
    // does not. Detected rather than configured so either artifact loads.
    private readonly bool _batched;
    private readonly Dictionary<string, float[]> _voiceCache = new(StringComparer.Ordinal);

    /// <summary>
    /// Load <c>kokoro.onnx</c> from <paramref name="onnxDir"/>. Voice packs are
    /// read lazily from <c>&lt;onnxDir&gt;/voices/&lt;name&gt;.bin</c> (produced by
    /// scripts/kokoro_export/export_voices.py).
    /// </summary>
    public Kokoro(string onnxDir, ExecutionProvider ep, SessionLoadObserver? onLoad = null)
    {
        // cudnn_conv_algo_search=DEFAULT. Every synthesis call is a different chunk length, so
        // ORT's EXHAUSTIVE default re-benchmarks the decoder's convs on each new shape and never
        // amortizes the tuning. Measured 1.47x end-to-end (ORT 1.29 / RTX 3090) with output at the
        // noise floor. See OrtSessionBuilder.AppendCuda and docs/kokoro_onnx_investigation.md Run 24.
        // kokoro_batched.onnx supersedes kokoro.onnx: it is faster at every batch size including
        // B=1, so prefer it whenever the directory has one. Falling back keeps older model
        // directories (and anyone's existing download) working unchanged.
        var batched = Path.Combine(onnxDir, "kokoro_batched.onnx");
        var modelPath = File.Exists(batched) ? batched : Path.Combine(onnxDir, "kokoro.onnx");
        _session = SessionLoader.LoadAndReport(
            modelPath, ep, onLoad, cudnnConvAlgoSearch: "DEFAULT");
        _voicesDir = Path.Combine(onnxDir, "voices");
        _batched = _session.InputMetadata.ContainsKey("input_lengths");
    }

    /// <summary>
    /// True when the loaded graph is <c>kokoro_batched.onnx</c> and
    /// <see cref="SynthesizeBatch"/> runs one ONNX call for the whole batch. False for the
    /// batch=1 <c>kokoro.onnx</c>, where it falls back to a loop (same output, no speed-up).
    /// </summary>
    public bool SupportsBatching => _batched;

    /// <summary>
    /// Synthesize 24 kHz mono float32 audio from a Kokoro-alphabet phoneme
    /// string. <paramref name="voice"/> names a voice pack (e.g. "af_heart").
    /// <paramref name="speed"/> is the speech-rate multiplier (1.0 = natural).
    /// </summary>
    public float[] Synthesize(string phonemes, string voice, float speed = 1.0f)
        => SynthesizeWithDurations(phonemes, voice, speed).Audio;

    /// <summary>
    /// Synthesize and also return the per-input-token predicted durations and the
    /// encoded token ids — needed for word-level alignment (see <see cref="KokoroTts"/>).
    /// <see cref="KokoroOutput.PredDur"/> is in duration units; one audio frame is
    /// <c>Audio.Length / Σ PredDur</c> samples (exactly 600 at 24 kHz).
    /// </summary>
    public KokoroOutput SynthesizeWithDurations(string phonemes, string voice, float speed = 1.0f)
    {
        if (string.IsNullOrEmpty(phonemes))
            return new KokoroOutput([], [], []);

        // The batched graph requires input_lengths and returns a leading batch axis, so the
        // single-item call has to go through the same path — a batch of one.
        if (_batched)
            return SynthesizeBatch([phonemes], voice, speed)[0];

        var inputIds = KokoroVocab.Encode(phonemes);
        // ref_s is indexed by phoneme-STRING length (KPipeline: pack[len(ps)-1]),
        // not the encoded id count. Count runes to match Python len() semantics.
        var refRow = Math.Clamp(CountRunes(phonemes) - 1, 0, VoiceRows - 1);

        var refS = LoadVoiceRow(voice, refRow);

        var idsT = new DenseTensor<long>(inputIds, [1, inputIds.Length]);
        var refT = new DenseTensor<float>(refS, [1, StyleDim]);
        var speedT = new DenseTensor<float>(new[] { speed }, [1]);

        using var outputs = _session.Run([
            NamedOnnxValue.CreateFromTensor("input_ids", idsT),
            NamedOnnxValue.CreateFromTensor("ref_s", refT),
            NamedOnnxValue.CreateFromTensor("speed", speedT),
        ]);
        var byName = outputs.ToDictionary(v => v.Name, v => v);
        var audio = byName["audio"].AsTensor<float>().ToArray();
        var predDur = byName["pred_dur"].AsTensor<long>().ToArray();
        return new KokoroOutput(audio, predDur, inputIds);
    }

    /// <summary>
    /// Synthesize several phoneme strings in one ONNX call. Each item gets its own style row
    /// (<c>ref_s</c> is indexed by that item's own phoneme-string length) and its own durations,
    /// so the result is item-for-item identical to calling
    /// <see cref="SynthesizeWithDurations"/> on each — see the fidelity note below.
    ///
    /// <para>Throughput saturates around 8-16 items (~2.5x on an RTX 3090) and VRAM is never the
    /// binding constraint, so there is no reason to go much wider. Sorting the batch by phoneme
    /// length reduces padding and therefore wasted compute; it does NOT affect fidelity, because
    /// the padding error is a step function — two frames of padding cost as much as two hundred.</para>
    ///
    /// <para>⚠ Fidelity: <c>pred_dur</c> is bit-identical to the solo render regardless of batch
    /// composition, so word alignment never shifts. The waveform differs by about as much as two
    /// solo renders of the same input differ from each other — Kokoro is nondeterministic (its
    /// vocoder draws noise every call), so that is the floor rather than a compromise.
    /// docs/kokoro_onnx_investigation.md Runs 26-36.</para>
    /// </summary>
    public IReadOnlyList<KokoroOutput> SynthesizeBatch(
        IReadOnlyList<string> phonemes, string voice, float speed = 1.0f)
    {
        ArgumentNullException.ThrowIfNull(phonemes);
        if (phonemes.Count == 0) return [];

        if (!_batched)
            return [.. phonemes.Select(p => SynthesizeWithDurations(p, voice, speed))];

        var ids = new long[phonemes.Count][];
        for (var i = 0; i < phonemes.Count; i++)
            ids[i] = string.IsNullOrEmpty(phonemes[i]) ? [] : KokoroVocab.Encode(phonemes[i]);

        // An empty item has no tokens and would make input_lengths 0, which the packed LSTMs
        // reject. Run the non-empty ones and splice the empties back in at their own indices.
        var live = Enumerable.Range(0, phonemes.Count).Where(i => ids[i].Length > 0).ToArray();
        if (live.Length == 0)
            return [.. phonemes.Select(_ => new KokoroOutput([], [], []))];

        var maxTokens = live.Max(i => ids[i].Length);
        var flat = new long[live.Length * maxTokens];
        var refFlat = new float[live.Length * StyleDim];
        var lengths = new long[live.Length];
        for (var b = 0; b < live.Length; b++)
        {
            var src = ids[live[b]];
            src.CopyTo(flat, b * maxTokens);           // remainder stays 0 = pad
            lengths[b] = src.Length;
            var refRow = Math.Clamp(CountRunes(phonemes[live[b]]) - 1, 0, VoiceRows - 1);
            LoadVoiceRow(voice, refRow).CopyTo(refFlat, b * StyleDim);
        }

        using var outputs = _session.Run([
            NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(flat, [live.Length, maxTokens])),
            NamedOnnxValue.CreateFromTensor("ref_s", new DenseTensor<float>(refFlat, [live.Length, StyleDim])),
            NamedOnnxValue.CreateFromTensor("speed", new DenseTensor<float>(new[] { speed }, [1])),
            NamedOnnxValue.CreateFromTensor("input_lengths", new DenseTensor<long>(lengths, [live.Length])),
        ]);
        var byName = outputs.ToDictionary(v => v.Name, v => v);
        var audio = byName["audio"].AsTensor<float>();
        var predDur = byName["pred_dur"].AsTensor<long>();

        var paddedSamples = audio.Dimensions[1];
        var results = new KokoroOutput[phonemes.Count];
        for (var i = 0; i < results.Length; i++) results[i] = new KokoroOutput([], [], []);
        for (var b = 0; b < live.Length; b++)
        {
            var n = (int)lengths[b];
            var dur = new long[n];
            long frames = 0;
            for (var t = 0; t < n; t++) { dur[t] = predDur[b, t]; frames += dur[t]; }

            // Each item is valid for its OWN frame count; the rest of the row is batch padding.
            var samples = (int)Math.Min(frames * SamplesPerFrame, paddedSamples);
            var clip = new float[samples];
            for (var k = 0; k < samples; k++) clip[k] = audio[b, k];
            results[live[b]] = new KokoroOutput(clip, dur, ids[live[b]]);
        }
        return results;
    }

    private static int CountRunes(string s)
    {
        var n = 0;
        foreach (var _ in s.EnumerateRunes()) n++;
        return n;
    }

    /// <summary>Return the 256-float style vector at <paramref name="row"/> of the named voice.</summary>
    private float[] LoadVoiceRow(string voice, int row)
    {
        if (!_voiceCache.TryGetValue(voice, out var pack))
        {
            var path = Path.Combine(_voicesDir, voice + ".bin");
            if (!File.Exists(path))
                throw new FileNotFoundException(
                    $"Kokoro voice '{voice}' not found at {path}. " +
                    "Export voices with scripts/kokoro_export/export_voices.py.", path);

            var bytes = File.ReadAllBytes(path);
            var expected = VoiceRows * StyleDim * sizeof(float);
            if (bytes.Length != expected)
                throw new InvalidDataException(
                    $"Voice '{voice}': expected {expected} bytes ({VoiceRows}×{StyleDim} f32), got {bytes.Length}.");

            pack = MemoryMarshal.Cast<byte, float>(bytes).ToArray();  // x64 LE
            _voiceCache[voice] = pack;
        }

        var slice = new float[StyleDim];
        Array.Copy(pack, row * StyleDim, slice, 0, StyleDim);
        return slice;
    }

    public void Dispose() => _session.Dispose();
}
