using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Vernacula.Base.Inference;
using Vernacula.Base.Models;

namespace Vernacula.Base.Alignment;

/// <summary>
/// NFA-style forced aligner backed by a NeMo CTC ASR model exported via
/// <c>scripts/nemo_export/export_nfa_ctc_to_onnx.py</c>. Loads the bundle,
/// runs the preprocessor + CTC encoder, executes
/// <see cref="CtcForcedAlignment.Align"/>, and groups sentencepiece
/// tokens into words by the <c>▁</c> word-boundary marker.
///
/// Bundle expected on disk (per the export script's contract):
///   nemo128.onnx              — log-mel preprocessor
///   ctc-model.onnx            — encoder + CTC log-softmax head
///   tokenizer.model           — sentencepiece model bytes
///   export-report.json        — frame_shift_seconds + encoder_subsampling
///                               + sample_rate metadata
///
/// Construction loads two ORT sessions and ~250 KB of SP model into
/// memory — non-trivial cost; instantiate once per process and reuse.
/// Not thread-safe per ORT's session semantics; one instance per
/// concurrent caller.
/// </summary>
public sealed class NemoNfaAligner : IForcedAligner
{
    private readonly InferenceSession _preproc;
    private readonly InferenceSession _ctc;
    private readonly SimpleSpTokenizer _tokenizer;
    private readonly int _vocabSize;     // V (excludes blank)
    private readonly int _blankId;       // == V by export convention
    private readonly double _secondsPerEncoderFrame;
    private readonly int _expectedSampleRate;

    // U+2581 LOWER ONE EIGHTH BLOCK — the sentencepiece word-boundary marker
    // that prefixes the first token of each word in a Llama/SP vocab.
    private const char SpWordBoundary = '▁';

    /// <summary>Load the bundle at <paramref name="bundleDir"/>.</summary>
    /// <param name="ep">Execution provider for the ORT sessions. Auto
    /// resolves to CUDA when available, then DirectML, then CPU.</param>
    public NemoNfaAligner(string bundleDir, ExecutionProvider ep = ExecutionProvider.Auto)
    {
        string preprocPath = Path.Combine(bundleDir, "nemo128.onnx");
        string ctcPath = Path.Combine(bundleDir, "ctc-model.onnx");
        string tokenizerPath = Path.Combine(bundleDir, "tokenizer.model");
        string reportPath = Path.Combine(bundleDir, "export-report.json");

        foreach (var p in new[] { preprocPath, ctcPath, tokenizerPath, reportPath })
            if (!File.Exists(p))
                throw new FileNotFoundException(
                    $"NFA bundle missing required file: {p}. "
                    + "Re-export via scripts/nemo_export/export_nfa_ctc_to_onnx.py.", p);

        _preproc = OrtSessionBuilder.CreateCachedSession(preprocPath, ep);
        _ctc = OrtSessionBuilder.CreateCachedSession(ctcPath, ep);

        // SimpleSpTokenizer over vocab.txt — stopgap for Microsoft.ML.Tokenizers
        // 2.0.0's unigram-SP loader bug on the NeMo FastConformer's SP model.
        // See SimpleSpTokenizer's class docs for the quality envelope.
        // (tokenizer.model is still in the bundle for a future drop-in when
        // a working unigram-SP loader lands.)
        _tokenizer = new SimpleSpTokenizer(Path.Combine(bundleDir, "vocab.txt"));
        _vocabSize = _tokenizer.VocabSize;
        _blankId = _vocabSize;

        var report = JsonDocument.Parse(File.ReadAllText(reportPath)).RootElement;

        // frame_shift_seconds and encoder_subsampling come from the
        // export-report. Combined, they give the seconds-per-encoder-frame
        // factor we use to convert frame indices → audio time.
        double frameShift = report.GetProperty("frame_shift_seconds").GetDouble();
        int subsampling = report.GetProperty("encoder_subsampling").GetInt32();
        _secondsPerEncoderFrame = frameShift * subsampling;

        _expectedSampleRate = report.TryGetProperty("sample_rate", out var srProp)
            ? srProp.GetInt32() : 16000;
    }

    public IReadOnlyList<WordTiming> Align(float[] audio16k, string text, string languageIso)
    {
        if (audio16k.Length == 0 || string.IsNullOrWhiteSpace(text))
            return Array.Empty<WordTiming>();

        var (logProbs, numEncoderFrames, vocabPlusBlank) = RunPreprocAndCtc(audio16k);

        // Tokenize the reference text. Greedy-LPM SP — ids correspond to
        // the columns of logProbs (modulo blank at index _blankId).
        var encoded = _tokenizer.Encode(text);
        if (encoded.Count == 0) return Array.Empty<WordTiming>();

        // CTC alignment can't handle a target with a blank in it.
        // Defensive filter; SP shouldn't emit the blank piece.
        var tokenIds = new List<int>(encoded.Count);
        var tokenStrings = new List<string>(encoded.Count);
        foreach (var (id, piece) in encoded)
        {
            if (id == _blankId) continue;
            tokenIds.Add(id);
            tokenStrings.Add(piece);
        }
        if (tokenIds.Count == 0) return Array.Empty<WordTiming>();

        // Feasibility check: target needs at least N frames (+1 per
        // same-token adjacency for the CTC blank between them).
        int minFrames = tokenIds.Count;
        for (int i = 1; i < tokenIds.Count; i++)
            if (tokenIds[i] == tokenIds[i - 1]) minFrames++;
        if (numEncoderFrames < minFrames)
        {
            // Audio is too short for the reference text. Return empty —
            // the caller's transcript was longer than the audio could
            // possibly contain. (We could throw, but empty matches the
            // interface's "best-effort" doc.)
            return Array.Empty<WordTiming>();
        }

        var tokenAlignments = CtcForcedAlignment.Align(
            logProbs, numEncoderFrames, vocabPlusBlank, tokenIds, _blankId);

        return GroupTokensIntoWords(tokenAlignments, tokenStrings, _secondsPerEncoderFrame);
    }

    /// <summary>
    /// Group token-level alignments into word-level by sentencepiece's
    /// <c>▁</c> word-boundary convention: a token starting with <c>▁</c>
    /// begins a new word; tokens without <c>▁</c> continue the current
    /// word. Word start = first token's start; word end = last token's end.
    /// Internal-static so Vernacula.Tests can exercise it without
    /// instantiating the full aligner (which needs the bundle on disk).
    /// </summary>
    internal static IReadOnlyList<WordTiming> GroupTokensIntoWords(
        IReadOnlyList<TokenAlignment> tokenAlignments,
        IReadOnlyList<string> tokenStrings,
        double secondsPerEncoderFrame)
    {
        var words = new List<WordTiming>();
        var currentPieces = new List<string>();
        int currentStartFrame = -1;
        int currentEndFrame = -1;

        void Flush()
        {
            if (currentPieces.Count == 0) return;
            // Join pieces, strip ALL word-boundary markers (just the leading
            // one in well-formed SP output, but defensive). The output is
            // the readable word.
            string joined = string.Concat(currentPieces).Replace(SpWordBoundary.ToString(), "");
            if (joined.Length > 0)
            {
                words.Add(new WordTiming(
                    Text: joined,
                    StartSeconds: currentStartFrame * secondsPerEncoderFrame,
                    EndSeconds: (currentEndFrame + 1) * secondsPerEncoderFrame));
            }
            currentPieces.Clear();
            currentStartFrame = -1;
            currentEndFrame = -1;
        }

        for (int k = 0; k < tokenAlignments.Count; k++)
        {
            var ta = tokenAlignments[k];
            string piece = tokenStrings[k];
            // SP marks word starts with leading ▁. The very first token
            // of a sentence may or may not have ▁ depending on
            // AddDummyPrefix; treat the first token as a word start
            // either way.
            bool startsWord = currentPieces.Count == 0 || piece.StartsWith(SpWordBoundary);
            if (startsWord) Flush();
            if (currentStartFrame < 0) currentStartFrame = ta.StartFrame;
            currentEndFrame = ta.EndFrame;
            currentPieces.Add(piece);
        }
        Flush();
        return words;
    }

    /// <summary>
    /// Run the preprocessor and CTC graph on the audio. Returns the
    /// CTC log-probs as a flat [T_enc * (V+1)] float[], plus T_enc and
    /// V+1 for the Viterbi helper.
    /// </summary>
    private (float[] logProbs, int numEncoderFrames, int vocabPlusBlank) RunPreprocAndCtc(float[] audio16k)
    {
        // Preprocessor: waveforms [1, samples] + waveforms_lens [1] →
        // features [1, 80, T_feat] + features_lens [1].
        var wavTensor = new DenseTensor<float>(audio16k, new[] { 1, audio16k.Length });
        var wavLens = new DenseTensor<long>(new long[] { audio16k.Length }, new[] { 1 });
        DenseTensor<float> features;
        DenseTensor<long> featuresLens;
        using (var preprocOut = _preproc.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("waveforms", wavTensor),
            NamedOnnxValue.CreateFromTensor("waveforms_lens", wavLens),
        }))
        {
            var outList = preprocOut.ToList();
            // The preprocessor returns features and features_lens in
            // declaration order — features is rank-3, features_lens is
            // rank-1. Materialize into fresh DenseTensors so the
            // disposable can be released.
            var featsRaw = outList[0].AsTensor<float>();
            features = new DenseTensor<float>(featsRaw.ToArray(), featsRaw.Dimensions.ToArray());
            var lensRaw = outList[1].AsTensor<long>();
            featuresLens = new DenseTensor<long>(lensRaw.ToArray(), lensRaw.Dimensions.ToArray());
        }

        // CTC graph: audio_signal [B, 80, T_feat] + length [B] →
        // logprobs [B, T_enc, V+1]. T_enc per batch = ceil(features_lens / subsampling),
        // but ORT computes it for us — we just read it from the output shape.
        using var ctcOut = _ctc.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("audio_signal", features),
            NamedOnnxValue.CreateFromTensor("length", featuresLens),
        });
        var logProbsTensor = ctcOut.First().AsTensor<float>();
        var shape = logProbsTensor.Dimensions.ToArray();
        // shape = [1, T_enc, V+1]
        int numEncoderFrames = shape[1];
        int vocabPlusBlank = shape[2];
        if (vocabPlusBlank != _vocabSize + 1)
            throw new InvalidOperationException(
                $"CTC output vocab dim ({vocabPlusBlank}) doesn't match expected V+1 ({_vocabSize + 1}). "
                + "Bundle's vocab.txt and ctc-model.onnx are out of sync.");
        return (logProbsTensor.ToArray(), numEncoderFrames, vocabPlusBlank);
    }

    public void Dispose()
    {
        _preproc.Dispose();
        _ctc.Dispose();
        // LlamaTokenizer doesn't implement IDisposable — its underlying
        // SP model is GC-managed.
    }
}
