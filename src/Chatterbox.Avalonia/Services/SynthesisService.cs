using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Chatterbox.App.Models;
using Chatterbox.Base;
using Chatterbox.Base.Markdown;
using NAudio.Wave;
using Vernacula.Base.Alignment;
using Vernacula.Base.Models;

namespace Chatterbox.App.Services;

/// <summary>
/// One-call synthesis + alignment for the reader UI. Mirrors what
/// Chatterbox.CLI's `--alignment-out` path does end-to-end, just
/// in-process and structured for the UI to consume:
/// (audio WAV path, alignment data) returned together.
///
/// Sessions are lazily loaded on the first call and reused across
/// subsequent invocations — the pipeline load is the big sticky cost
/// (~5-10 s for Chatterbox + ~1 s for NFA), and rebuilding it per
/// click would make the UI feel slow.
/// </summary>
public sealed class SynthesisService : IDisposable
{
    private readonly string _onnxBundleDir;
    private readonly string? _nfaBundleDir;
    private readonly string? _tokenizerJsonPath;
    private readonly ExecutionProvider _ep;

    private ChatterboxPipeline? _pipeline;
    private NemoNfaAligner? _aligner;
    private readonly object _gate = new();

    /// <param name="onnxBundleDir">Path to the Chatterbox ONNX bundle
    /// (output of scripts/chatterbox_export/export_chatterbox_to_onnx.py).</param>
    /// <param name="nfaBundleDir">Optional NFA bundle path. When null,
    /// synthesis runs without producing alignment data (the returned
    /// <see cref="AlignmentSidecar.Words"/> stays empty).</param>
    /// <param name="tokenizerJsonPath">Optional override for the
    /// Chatterbox tokenizer.json; null = auto-locate from HF hub cache.</param>
    public SynthesisService(
        string onnxBundleDir,
        string? nfaBundleDir = null,
        string? tokenizerJsonPath = null,
        ExecutionProvider ep = ExecutionProvider.Auto)
    {
        _onnxBundleDir = onnxBundleDir;
        _nfaBundleDir = nfaBundleDir;
        _tokenizerJsonPath = tokenizerJsonPath;
        _ep = ep;
    }

    public sealed record SynthesisResult(string AudioPath, AlignmentSidecar Alignment);

    /// <summary>
    /// Synthesize <paramref name="text"/> in the voice of
    /// <paramref name="voicePath"/>, write a WAV to <paramref name="outWavPath"/>,
    /// and (when an NFA bundle is configured) produce per-word audio
    /// timings. Runs on whatever thread you call it from — the UI
    /// should call this via Task.Run to keep the dispatcher responsive.
    /// </summary>
    public SynthesisResult Synthesize(
        string voicePath,
        string text,
        string outWavPath,
        CancellationToken cancellationToken = default)
    {
        EnsureLoaded();
        cancellationToken.ThrowIfCancellationRequested();

        // Markdown-extract when text looks like markdown (loose heuristic:
        // any of the common block markers). For the MVP this lets users
        // paste markdown directly into the UI's text box; richer file-vs-
        // string handling can come later.
        var preparedText = LooksLikeMarkdown(text)
            ? MarkdownTextExtractor.Extract(text).Text
            : text;
        if (string.IsNullOrWhiteSpace(preparedText))
            throw new InvalidOperationException("Text is empty after markdown extraction.");

        var chunks = ParagraphChunker.Chunk(preparedText);
        var spk = _pipeline!.Embedder.Embed(voicePath);
        cancellationToken.ThrowIfCancellationRequested();

        float[][] chunkAudios;
        if (chunks.Count <= 1)
        {
            var tokenIds = _pipeline.Tokenizer!.WrapForLm(preparedText);
            var lmResult = _pipeline.Lm.Generate(spk.CondEmb, tokenIds);
            var speechTokens = lmResult.BuildSpeechTokens(spk.AudioTokens);
            var samples = _pipeline.Vocoder.Synthesize(
                speechTokens, spk.SpeakerEmbeddings, spk.SpeakerFeatures);
            chunkAudios = new[] { samples };
        }
        else
        {
            var tokensPerChunk = chunks.Select(c => _pipeline.Tokenizer!.WrapForLm(c)).ToArray();
            var synth = new ChunkedSynthesizer(_pipeline);
            var result = synth.Synthesize(spk, tokensPerChunk);
            chunkAudios = result.Waveforms.ToArray();
        }
        cancellationToken.ThrowIfCancellationRequested();

        // Concatenate + write WAV (24 kHz mono float32 — same as CLI).
        int totalSamples = chunkAudios.Sum(w => w.Length);
        var allSamples = new float[totalSamples];
        int off = 0;
        foreach (var w in chunkAudios)
        {
            Array.Copy(w, 0, allSamples, off, w.Length);
            off += w.Length;
        }
        var fmt = WaveFormat.CreateIeeeFloatWaveFormat(ChatterboxConstants.S3GenSr, 1);
        using (var writer = new WaveFileWriter(outWavPath, fmt))
            writer.WriteSamples(allSamples, 0, allSamples.Length);

        // Alignment. When no NFA bundle is configured, return an empty
        // sidecar (audio-only mode) — UI still shows playback, just
        // without word highlighting.
        var sidecar = new AlignmentSidecar
        {
            AudioPath = outWavPath,
            SampleRate = ChatterboxConstants.S3GenSr,
            AudioDurationSeconds = totalSamples / (double)ChatterboxConstants.S3GenSr,
            Aligner = _aligner is null ? "none" : "nemo_nfa",
        };
        if (_aligner is not null)
        {
            int sampleOffset = 0;
            for (int i = 0; i < chunkAudios.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var chunkAudio24k = chunkAudios[i];
                double chunkStartSec = sampleOffset / (double)ChatterboxConstants.S3GenSr;
                double chunkEndSec = (sampleOffset + chunkAudio24k.Length) / (double)ChatterboxConstants.S3GenSr;
                sampleOffset += chunkAudio24k.Length;

                var chunkAudio16k = Vernacula.Base.AudioUtils.AudioTo16000Mono(
                    chunkAudio24k, ChatterboxConstants.S3GenSr, channels: 1);
                var words = _aligner.Align(chunkAudio16k, chunks[i], "en");

                sidecar.Chunks.Add(new ChunkRecord
                {
                    Index = i,
                    AudioStartSeconds = chunkStartSec,
                    AudioEndSeconds = chunkEndSec,
                    Text = chunks[i],
                    WordCount = words.Count,
                });
                foreach (var w in words)
                {
                    sidecar.Words.Add(new AlignedWord
                    {
                        Text = w.Text,
                        StartSeconds = chunkStartSec + w.StartSeconds,
                        EndSeconds = chunkStartSec + w.EndSeconds,
                        ChunkIndex = i,
                    });
                }
            }
        }
        return new SynthesisResult(outWavPath, sidecar);
    }

    private void EnsureLoaded()
    {
        if (_pipeline is not null) return;
        lock (_gate)
        {
            if (_pipeline is not null) return;
            _pipeline = new ChatterboxPipeline(_onnxBundleDir, _ep, _tokenizerJsonPath);
            if (_nfaBundleDir is not null && Directory.Exists(_nfaBundleDir))
                _aligner = new NemoNfaAligner(_nfaBundleDir, _ep);
        }
    }

    private static bool LooksLikeMarkdown(string text)
        => text.Contains("\n#") || text.Contains("\n- ") || text.Contains("\n* ")
           || text.Contains("\n```") || text.StartsWith("#") || text.StartsWith("- ");

    public void Dispose()
    {
        _aligner?.Dispose();
        _pipeline?.Dispose();
    }
}
