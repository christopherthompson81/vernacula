using System;

namespace Vernacula.Base;

/// <summary>
/// The device-memory model for VibeVoice-ASR-Streaming, in one place.
///
/// <see cref="VibeVoiceStreamingAsr"/> applies it to the package it has loaded, to decide how
/// large a KV cache to allocate. The settings window applies it to the published packages'
/// known shapes, to tell someone what a card will manage before anything is downloaded. Those
/// two answers have to agree, and they did not while each carried its own copy of the
/// arithmetic (issue #150 review).
/// </summary>
public static class VibeVoiceStreamingBudget
{
    /// <summary>
    /// Cache positions consumed per second of audio. Each hop contributes its frames, the two
    /// speech markers, the chunk-end token and the text generated for it; only the last varies,
    /// with speech density. Measured at ~16.0 (Run 14), against a floor of ~9.9 for audio with
    /// no speech at all.
    /// </summary>
    public const double PositionsPerSecond = 16.0;

    /// <summary>
    /// Positions a run spends before the audio: the prompt, its hotwords, and the rounding on
    /// the last partial window. Small, but it is the difference between promising a length and
    /// then refusing it.
    /// </summary>
    public const int PromptPositionSlack = 512;

    /// <summary>
    /// Device memory one cached position costs: a key and a value, every layer, float16.
    /// 28 KiB for the published 1.5B, 56 KiB for the 7B.
    /// </summary>
    public static long KvBytesPerPosition(int numLayers, int numKvHeads, int headDim) =>
        (long)numLayers * 2 * numKvHeads * headDim * 2;

    /// <summary>
    /// Device memory a run needs beyond the weights and the cache: ONNX Runtime's CUDA arena,
    /// the encoder's activations for one window, and the logits buffer. Measured against the
    /// published packages at 1.39 GiB (1.5B, hidden 1536) and 2.36 GiB (7B, hidden 3584) —
    /// Run 34 — so it scales with the decoder's hidden size, and this rounds up on both rather
    /// than sailing close.
    /// </summary>
    public static long WorkingSetBytes(int hiddenSize) =>
        (1L << 30) + (long)hiddenSize * 512 * 1024;

    /// <summary>
    /// Minutes of audio a run can actually take: the lesser of what
    /// <paramref name="freeBytes"/> pays for once the weights and working set are covered, and
    /// what the package's own context ceiling allows. Negative when the model does not fit at
    /// all, so callers can tell "no room for audio" from "no room".
    ///
    /// Both limits are net of <see cref="PromptPositionSlack"/> and rounded down to a whole
    /// minute, because the planner rounds the other way — it takes the ceiling of the estimate
    /// for the length it is handed. A figure that used the last position exactly would come
    /// back refused by one, which is the bug this shape exists to prevent.
    /// </summary>
    public static double MinutesThatFit(
        long freeBytes, long weightBytes, int hiddenSize, long kvBytesPerPosition,
        int ceilingPositions)
    {
        double byMemory = Math.Floor(
                              (freeBytes - weightBytes - WorkingSetBytes(hiddenSize))
                              / (double)kvBytesPerPosition)
                        - PromptPositionSlack;
        return Math.Floor(Math.Min(byMemory, CeilingPositions(ceilingPositions))
                          / PositionsPerSecond / 60);
    }

    /// <summary>Minutes the package's context ceiling allows, whatever the card can hold.</summary>
    public static double CeilingMinutes(int ceilingPositions) =>
        Math.Floor(CeilingPositions(ceilingPositions) / PositionsPerSecond / 60);

    private static double CeilingPositions(int ceilingPositions) =>
        ceilingPositions - PromptPositionSlack;
}
