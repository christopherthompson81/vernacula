using Chatterbox.Base.Tokenization;
using Vernacula.Base.Models;

namespace Chatterbox.Base;

/// <summary>
/// End-to-end Chatterbox TTS pipeline. Loads + owns the
/// <see cref="SpeakerEmbedder"/>, <see cref="AcousticLM"/>, and
/// <see cref="Vocoder"/> built from the ONNX bundle in
/// <c>onnxDir</c>, plus an <see cref="EnTokenizer"/> for text input.
///
/// Typical use:
/// <code>
/// using var pipeline = new ChatterboxPipeline("/path/to/cb_dyn5", ExecutionProvider.Auto);
/// var wav = pipeline.Synthesize(voicePath: "ref.wav", text: "Hello world.");
/// // write `wav` as a 24 kHz mono float32 WAV
/// </code>
///
/// Designed for the simple case (one voice, one text → one waveform).
/// Callers that need finer control (separate speaker-embedding cache, custom
/// LM rollout parameters, streaming output) should compose the underlying
/// classes directly instead of going through this façade.
/// </summary>
public sealed class ChatterboxPipeline : IDisposable
{
    public SpeakerEmbedder Embedder { get; }
    public AcousticLM Lm { get; }
    public Vocoder Vocoder { get; }
    public EnTokenizer? Tokenizer { get; }

    /// <summary>
    /// Load all four (or six) sessions from <paramref name="onnxDir"/>.
    /// If <paramref name="tokenizerJsonPath"/> is null, attempts to locate
    /// <c>tokenizer.json</c> in the HuggingFace hub cache at
    /// <c>~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/*/</c>.
    /// If neither succeeds, <see cref="Tokenizer"/> is null and only
    /// <see cref="Synthesize(string, long[])"/> (pre-tokenized) will work.
    /// </summary>
    /// <param name="onLoad">Optional callback fired once per ONNX session
    /// loaded (5 times for the Merged-vocoder layout, 7 times for Split).
    /// Use for surfacing per-session timing / cache state to the user.</param>
    public ChatterboxPipeline(string onnxDir, ExecutionProvider ep,
        string? tokenizerJsonPath = null, SessionLoadObserver? onLoad = null)
    {
        Embedder = new SpeakerEmbedder(Path.Combine(onnxDir, "speech_encoder.onnx"), ep, onLoad);
        Lm = new AcousticLM(
            Path.Combine(onnxDir, "embed_tokens.onnx"),
            Path.Combine(onnxDir, "language_model.onnx"),
            ep, onLoad);
        Vocoder = new Vocoder(onnxDir, ep, onLoad);

        var resolvedTok = tokenizerJsonPath ?? LocateCachedTokenizerJson();
        Tokenizer = resolvedTok is not null ? new EnTokenizer(resolvedTok) : null;
    }

    /// <summary>
    /// Synthesize speech matching <paramref name="text"/> in the voice of
    /// <paramref name="voicePath"/>. Requires <see cref="Tokenizer"/> to
    /// be non-null (a tokenizer.json was found or supplied).
    /// </summary>
    public float[] Synthesize(string voicePath, string text,
        bool useIoBinding = true,
        float exaggeration = ChatterboxConstants.DefaultExaggeration,
        int maxSteps = ChatterboxConstants.DefaultMaxLmSteps)
    {
        if (Tokenizer is null)
            throw new InvalidOperationException(
                "No tokenizer available. Pass `tokenizerJsonPath:` to the pipeline ctor, " +
                "or use Synthesize(voicePath, textTokenIds) with pre-tokenized ids.");
        var tokenIds = Tokenizer.WrapForLm(text);
        return Synthesize(voicePath, tokenIds, useIoBinding, exaggeration, maxSteps);
    }

    /// <summary>
    /// Synthesize speech from pre-tokenized LM input ids (the form
    /// <see cref="EnTokenizer.WrapForLm"/> emits). Avoids needing a
    /// tokenizer; useful for tests and for non-English models.
    /// </summary>
    public float[] Synthesize(string voicePath, long[] textTokenIds,
        bool useIoBinding = true,
        float exaggeration = ChatterboxConstants.DefaultExaggeration,
        int maxSteps = ChatterboxConstants.DefaultMaxLmSteps)
    {
        var spk = Embedder.Embed(voicePath);
        var lmResult = Lm.Generate(spk.CondEmb, textTokenIds,
            useIoBinding: useIoBinding,
            exaggeration: exaggeration,
            maxSteps: maxSteps);
        var speechTokens = lmResult.BuildSpeechTokens(spk.AudioTokens);
        return Vocoder.Synthesize(speechTokens, spk.SpeakerEmbeddings, spk.SpeakerFeatures);
    }

    /// <summary>
    /// Best-effort lookup of the chatterbox <c>tokenizer.json</c> in the
    /// standard HuggingFace hub cache. Returns null if not present; the
    /// caller can pass the path explicitly to the ctor instead.
    /// </summary>
    public static string? LocateCachedTokenizerJson()
    {
        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        var snapshotsDir = Path.Combine(home,
            ".cache", "huggingface", "hub", "models--ResembleAI--chatterbox", "snapshots");
        if (!Directory.Exists(snapshotsDir)) return null;
        foreach (var snap in Directory.EnumerateDirectories(snapshotsDir))
        {
            var p = Path.Combine(snap, "tokenizer.json");
            if (File.Exists(p)) return p;
        }
        return null;
    }

    public void Dispose()
    {
        Vocoder.Dispose();
        Lm.Dispose();
        Embedder.Dispose();
    }
}
