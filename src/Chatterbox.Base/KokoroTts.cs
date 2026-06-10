using Vernacula.Base.Models;
using Vernacula.Phonemizer;
using Vernacula.Phonemizer.Data;
using Vernacula.Phonemizer.Types;

namespace Chatterbox.Base;

/// <summary>
/// End-to-end Kokoro-82M text-to-speech: text → phonemes → audio. Composes the
/// pure-C# espeak-ng port (<see cref="Phonemize"/>), the Kokoro render format
/// (<see cref="KokoroFormat"/>), and the ONNX inference path (<see cref="Kokoro"/>).
///
/// G2P uses the misaki-style frontend for English: phonemize to IPA, then render
/// to Kokoro's alphabet. The phonemizer needs its language data directory (the
/// <c>data/</c> tree shipped with the espeak-ng-portable submodule) — pass its
/// path as <c>phonemizerDataDir</c>; en/en-gb subfolders are loaded from there.
///
/// Not thread-safe (wraps <see cref="Kokoro"/> / ORT). One instance per caller.
/// </summary>
public sealed class KokoroTts : IDisposable
{
    private readonly Kokoro _kokoro;
    private readonly string _dataDir;
    private readonly Language _enUs;
    private Language? _enGb;

    /// <param name="onnxDir">Directory holding kokoro.onnx and voices/.</param>
    /// <param name="phonemizerDataDir">The espeak-ng-portable <c>data/</c> directory
    /// (contains <c>en/</c>, <c>en-gb/</c>, …).</param>
    public KokoroTts(string onnxDir, string phonemizerDataDir, ExecutionProvider ep,
                     SessionLoadObserver? onLoad = null)
    {
        _kokoro = new Kokoro(onnxDir, ep, onLoad);
        _dataDir = phonemizerDataDir;
        _enUs = LanguageLoader.Load("en", Path.Combine(_dataDir, "en"));
    }

    /// <summary>Output sample rate (24 kHz).</summary>
    public int SampleRate => Kokoro.SampleRate;

    /// <summary>
    /// Synthesize 24 kHz mono float32 audio from <paramref name="text"/> using the
    /// given <paramref name="voice"/> (e.g. "af_heart"). Set <paramref name="british"/>
    /// for en-gb pronunciation + voices (bf_*/bm_*).
    /// </summary>
    public float[] Speak(string text, string voice, float speed = 1.0f, bool british = false)
        => _kokoro.Synthesize(ToPhonemes(text, british), voice, speed);

    /// <summary>
    /// Text → Kokoro-alphabet phoneme string, without running the vocoder. Useful
    /// for inspection, caching, or feeding <see cref="Kokoro.Synthesize"/> directly.
    /// </summary>
    public string ToPhonemes(string text, bool british = false)
    {
        var lang = british ? (_enGb ??= LanguageLoader.Load("en-gb", Path.Combine(_dataDir, "en-gb"))) : _enUs;
        var ipa = Phonemize.Run(text, lang);
        return KokoroFormat.Render(ipa, british);
    }

    public void Dispose() => _kokoro.Dispose();
}
