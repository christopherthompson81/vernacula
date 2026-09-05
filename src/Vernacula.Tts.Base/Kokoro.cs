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

    private readonly InferenceSession _session;
    private readonly string _voicesDir;
    private readonly Dictionary<string, float[]> _voiceCache = new(StringComparer.Ordinal);

    /// <summary>
    /// Load <c>kokoro.onnx</c> from <paramref name="onnxDir"/>. Voice packs are
    /// read lazily from <c>&lt;onnxDir&gt;/voices/&lt;name&gt;.bin</c> (produced by
    /// scripts/kokoro_export/export_voices.py).
    /// </summary>
    public Kokoro(string onnxDir, ExecutionProvider ep, SessionLoadObserver? onLoad = null)
    {
        _session = SessionLoader.LoadAndReport(Path.Combine(onnxDir, "kokoro.onnx"), ep, onLoad);
        _voicesDir = Path.Combine(onnxDir, "voices");
    }

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
