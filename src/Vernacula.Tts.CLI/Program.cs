// vernacula-tts — IPA-native text-to-speech.
//
// Joins the two halves of the IPA pipeline: vernacula-phonemizer turns text in any of its 192
// routed languages into canonical IPA, and the IPA fine-tune of OmniVoice (a LoRA diff folded onto
// the base transformer at load) renders that IPA as speech, optionally in a cloned voice.
//
// The model is conditioned as language-agnostic — the style prefix is <|lang_start|>None<|lang_end|>
// and the IPA stream carries everything, which is what the fine-tune was trained and
// listen-accepted on (scripts/omnivoice_ipa/gen_accept_test.py). So --lang here selects the
// PHONEMIZER, never the model.
//
// Usage:
//   vernacula-tts --lang en --text "Hello world" --out out.wav \
//                 --onnx-dir scripts/omnivoice_export/onnx \
//                 --model-dir /path/to/k2-fsa-OmniVoice
//   vernacula-tts --lang cy --text-file page.md --voice ref.wav --ref-text "..." --out out.wav
//   vernacula-tts --ipa --text "hɛlˈoʊ wˈɜːld." --out out.wav

using System.Diagnostics;
using System.Globalization;
using Chatterbox.Base;
using Chatterbox.Base.Markdown;
using NAudio.Wave;
using Vernacula.Base;
using Vernacula.Base.Models;

namespace Vernacula.Tts.CLI;

internal static class Program
{
    /// <summary>Duration estimate (in audio tokens, 25 ≈ 1 s) below which a no-reference
    /// generation is out of the fine-tune's training distribution. 125 ≈ 5 s: the corpus's
    /// 1st percentile is 4.8 s and its median is 12 s.</summary>
    private const int ShortInputTokens = 125;

    private static async Task<int> Main(string[] args)
    {
        // ── Argument parsing ──────────────────────────────────────────────────────────────────
        string? text = null, textFile = null, outPath = null;
        string? lang = null, refLang = null;
        string? voicePath = null, refText = null;
        string? onnxDir = Environment.GetEnvironmentVariable("OMNIVOICE_ONNX_DIR");
        string? modelDir = Environment.GetEnvironmentVariable("OMNIVOICE_MODEL_DIR");
        string? tokenizerJson = null, dataDir = null;
        string? diffPath = null, transformerFile = null;
        bool noDiff = false, rawIpa = false, printIpa = false, verbose = false;
        string ep = "auto";
        int numStep = 32;
        int? targetTokens = null;

        string Next(string flag, int i)
        {
            if (i + 1 >= args.Length) throw new ArgumentException($"{flag} requires a value");
            return args[i + 1];
        }

        try
        {
            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--text":            text            = Next(args[i], i); i++; break;
                    case "--text-file":       textFile        = Next(args[i], i); i++; break;
                    case "--out":             outPath         = Next(args[i], i); i++; break;
                    case "--lang":            lang            = Next(args[i], i); i++; break;
                    case "--ref-lang":        refLang         = Next(args[i], i); i++; break;
                    case "--voice":           voicePath       = Next(args[i], i); i++; break;
                    case "--ref-text":        refText         = Next(args[i], i); i++; break;
                    case "--onnx-dir":        onnxDir         = Next(args[i], i); i++; break;
                    case "--model-dir":       modelDir        = Next(args[i], i); i++; break;
                    case "--tokenizer-json":  tokenizerJson   = Next(args[i], i); i++; break;
                    case "--data-dir":        dataDir         = Next(args[i], i); i++; break;
                    case "--diff":            diffPath        = Next(args[i], i); i++; break;
                    case "--transformer-file": transformerFile = Next(args[i], i); i++; break;
                    case "--ep":              ep              = Next(args[i], i).ToLowerInvariant(); i++; break;
                    case "--no-diff":         noDiff          = true; break;
                    case "--ipa":             rawIpa          = true; break;
                    case "--print-ipa":       printIpa        = true; break;
                    case "--verbose" or "-v": verbose         = true; break;
                    case "--num-step":
                        if (!int.TryParse(Next(args[i], i), NumberStyles.Integer, CultureInfo.InvariantCulture,
                                out numStep) || numStep < 1)
                            throw new ArgumentException("--num-step expects a positive integer (default 32).");
                        i++; break;
                    case "--target-tokens":
                        if (!int.TryParse(Next(args[i], i), NumberStyles.Integer, CultureInfo.InvariantCulture,
                                out int tt) || tt < 1)
                            throw new ArgumentException("--target-tokens expects a positive integer.");
                        targetTokens = tt; i++; break;
                    case "--help" or "-h":
                        PrintUsage(); return 0;
                    default:
                        Console.Error.WriteLine($"Unknown arg: {args[i]}");
                        PrintUsage();
                        return 2;
                }
            }
        }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 2;
        }

        // ── Validate ──────────────────────────────────────────────────────────────────────────
        if (text is null == (textFile is null))
        {
            Console.Error.WriteLine("Give exactly one of --text or --text-file.");
            return 2;
        }
        if (outPath is null) { Console.Error.WriteLine("--out <wav> is required."); return 2; }
        if (onnxDir is null)
        {
            Console.Error.WriteLine("--onnx-dir <dir> is required (or set OMNIVOICE_ONNX_DIR). It holds "
                + $"{OmniVoice.TransformerFile} (+ .onnx.data), {OmniVoice.EncoderFile}, {OmniVoice.DecoderFile}.");
            return 2;
        }
        if (!Directory.Exists(onnxDir)) { Console.Error.WriteLine($"--onnx-dir not found: {onnxDir}"); return 1; }
        if (!rawIpa && lang is null)
        {
            Console.Error.WriteLine("--lang <code> is required (the phonemizer language, e.g. en, cy, cmn). "
                + "Pass --ipa instead if --text is already IPA.");
            return 2;
        }
        if (rawIpa && lang is not null)
        {
            Console.Error.WriteLine("--ipa and --lang are mutually exclusive: --ipa means the text is already "
                + "phonemized, so there is no phonemizer language to select.");
            return 2;
        }
        if (textFile is not null && !File.Exists(textFile))
        {
            Console.Error.WriteLine($"--text-file not found: {textFile}"); return 1;
        }
        if (voicePath is not null && !File.Exists(voicePath))
        {
            Console.Error.WriteLine($"--voice not found: {voicePath}"); return 1;
        }
        if (voicePath is not null && refText is null)
        {
            Console.Error.WriteLine("Voice cloning needs the reference transcript: pass --ref-text \"...\". "
                + "(With --ipa it must itself be IPA; otherwise it is phonemized with --ref-lang, "
                + "defaulting to --lang.)");
            return 2;
        }
        if (refText is not null && voicePath is null)
        {
            Console.Error.WriteLine("--ref-text describes --voice; pass a reference WAV or drop --ref-text.");
            return 2;
        }

        // Qwen3 tokenizer: --tokenizer-json, else tokenizer.json beside the graphs, else in --model-dir.
        tokenizerJson ??= FirstExisting(
            Path.Combine(onnxDir, "tokenizer.json"),
            modelDir is null ? null : Path.Combine(modelDir, "tokenizer.json"));
        if (tokenizerJson is null)
        {
            Console.Error.WriteLine("Qwen3 tokenizer.json not found. Pass --model-dir <k2-fsa-OmniVoice "
                + "snapshot> (or OMNIVOICE_MODEL_DIR), or --tokenizer-json <path>.");
            return 2;
        }
        if (!File.Exists(tokenizerJson))
        {
            Console.Error.WriteLine($"--tokenizer-json not found: {tokenizerJson}"); return 1;
        }

        // ── The IPA fine-tune diff ────────────────────────────────────────────────────────────
        // Default it ON when the versioned diff sits beside the graphs: without it this is stock
        // OmniVoice, which reads IPA as if it were orthography and produces confident nonsense.
        // That failure is only audible, so a silent fallback is the wrong default.
        if (!noDiff)
        {
            diffPath ??= FirstExisting(Path.Combine(onnxDir, IpaFineTune.DefaultDiffFile));
            if (diffPath is null)
            {
                Console.Error.WriteLine($"IPA fine-tune diff not found: {Path.Combine(onnxDir, IpaFineTune.DefaultDiffFile)}\n"
                    + "  Pass --diff <path>, or --no-diff to run the base (orthographic) model knowing that\n"
                    + "  IPA input will not be interpreted as phonemes.");
                return 1;
            }
        }
        else if (diffPath is not null)
        {
            Console.Error.WriteLine("--diff and --no-diff are mutually exclusive."); return 2;
        }
        if (diffPath is not null && !File.Exists(diffPath))
        {
            Console.Error.WriteLine($"--diff not found: {diffPath}"); return 1;
        }

        // ⚠ Reject an unrecognised value rather than falling through to Auto. `--ep cud` or
        // `--ep gpu` would otherwise silently run somewhere the caller did not ask for — and the
        // gap between providers here is ~20x, so it reads as "this is just slow".
        // Chatterbox.CLI errors on the same input; match it.
        ExecutionProvider epEnum;
        switch (ep)
        {
            case "cpu": epEnum = ExecutionProvider.Cpu; break;
            case "cuda": epEnum = ExecutionProvider.Cuda; break;
            case "auto": epEnum = ExecutionProvider.Auto; break;
            default:
                Console.Error.WriteLine($"--ep must be cpu, cuda or auto (got \"{ep}\").");
                return 2;
        }

        // ⚠ THE DIFF MUST BE APPLIED TO THE GENUINE BASE, and nothing here can check that for you.
        // The diff is a LoRA plus absolute replacement embedding rows; applied to weights that are
        // not the base OmniVoice checkpoint it produces a plausible-looking model that is quietly
        // wrong. Measured: the per-row delta distributions for a right and a wrong base overlap
        // (median 0.0086 vs 0.0124), so there is no self-check to derive from the diff's contents.
        // Point --transformer-file at the base whose weights match the k2-fsa/OmniVoice checkpoint.

        // ── Text → IPA ────────────────────────────────────────────────────────────────────────
        string sourceText;
        if (text is not null) sourceText = text;
        else
        {
            var raw = File.ReadAllText(textFile!);
            var ext = Path.GetExtension(textFile!).ToLowerInvariant();
            sourceText = ext is ".md" or ".markdown" ? MarkdownTextExtractor.Extract(raw).Text : raw;
        }
        if (string.IsNullOrWhiteSpace(sourceText)) { Console.Error.WriteLine("Text is empty."); return 1; }

        string ipaText, ipaRefText;
        if (rawIpa)
        {
            ipaText = sourceText.Trim();
            ipaRefText = refText?.Trim() ?? "";
        }
        else
        {
            if (PhonemizerData.Resolve(dataDir) is not { } root)
            {
                Console.Error.WriteLine(PhonemizerData.NotFoundMessage());
                return 1;
            }
            if (verbose) Console.WriteLine($"phonemizer data: {root}");
            if (!IpaFineTune.TrainedLanguages.Contains(lang!))
                Console.Error.WriteLine($"NOTE: '{lang}' was not in the {IpaFineTune.TrainedLanguages.Count}-language "
                    + "IPA fine-tune corpus. It renders from phones the model already holds — that is the point of "
                    + "an IPA-conditioned model — but the result is extrapolated, and prosody especially so.");

            try
            {
                // PhonemizeAsync is the best-output entry: it routes through each language's neural
                // model where one exists (English's BiLSTM, the taggers, the abjad vowel restorers).
                ipaText = (await global::Vernacula.Phonemizer.Phonemizer.PhonemizeAsync(sourceText, lang!)).Trim();
                ipaRefText = refText is null
                    ? ""
                    : (await global::Vernacula.Phonemizer.Phonemizer.PhonemizeAsync(refText, refLang ?? lang!)).Trim();
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Phonemization failed for --lang {lang}: {ex.Message}");
                return 1;
            }
        }
        if (string.IsNullOrWhiteSpace(ipaText)) { Console.Error.WriteLine("Phonemized text is empty."); return 1; }

        // Parity with create_voice_clone_prompt: the reference transcript gets add_punctuation, and
        // the punctuated form feeds BOTH the duration estimate and the combined text. The IPA
        // stream keeps its , . ! ? — they carry prosody, and the fine-tune trained on them.
        string? condRefText = refText is null ? null : OmniVoiceTextPrep.AddPunctuation(ipaRefText);

        if (printIpa || verbose)
        {
            if (condRefText is not null) Console.WriteLine($"ref IPA: {condRefText}");
            Console.WriteLine($"IPA:     {ipaText}");
        }

        // ── Model ─────────────────────────────────────────────────────────────────────────────
        SessionLoadObserver? onLoad = verbose
            ? e => Console.WriteLine($"  {e.FileName}: {e.ElapsedMs} ms  ep={(e.UsedCuda ? "cuda" : "cpu/dml")}")
            : null;

        string mode = voicePath is not null ? "clone" : "auto";
        Console.WriteLine($"vernacula-tts: mode={mode}  lang={(rawIpa ? "(raw IPA)" : lang)}  steps={numStep}  ep={ep}"
            + $"  diff={(diffPath is null ? "none" : Path.GetFileName(diffPath))}");

        var sw = Stopwatch.StartNew();
        using var tts = new OmniVoiceTts(onnxDir, tokenizerJson, epEnum, onLoad, transformerFile, diffPath);
        Console.WriteLine($"load: {sw.ElapsedMilliseconds} ms");

        long[,]? refCodes = null;
        int refTokens = 25;
        float? refRms = null;   // pre-boost reference RMS, for the output un-boost below
        if (voicePath is not null)
        {
            float[] refWav = Load24kMono(voicePath);
            refRms = Rms(refWav);
            refCodes = tts.EncodeReference(refWav);
            refTokens = refCodes.GetLength(1);
            Console.WriteLine($"reference: {refWav.Length / (double)OmniVoiceTts.SampleRate:f2}s -> {refTokens} codes");
        }

        // Duration is estimated on the IPA, not the orthography — deliberately. The estimator is a
        // ratio (target script-weight / reference script-weight × reference tokens), so it is only
        // self-consistent when both sides are the same representation, and IPA-on-both is the
        // pacing the fine-tune was listen-accepted with (gen_accept_test.py: no duration forcing).
        int target = targetTokens
            ?? OmniVoiceDuration.EstimateTargetTokens(ipaText, condRefText, voicePath is not null ? refTokens : null);
        if (verbose) Console.WriteLine($"target tokens: {target}");
        if (target > 1500)
            Console.Error.WriteLine($"WARNING: estimated {target} tokens (~{target / 25}s) in one shot. Long-form "
                + "chunking isn't implemented for this backend yet; output may degrade. Split the input for now.");
        // The other end, and the more common trap — but it takes BOTH conditions, so the warning does too.
        //
        // The v6 fine-tune trained on FLEURS read sentences: median 12.0 s, 1st percentile 4.8 s, only
        // 0.21% of 268,165 utterances under 3 s. Short input is genuinely out of distribution, and it
        // does not degrade gracefully — it comes out as noise, deterministically (greedy decode,
        // temperature 0, so re-running reproduces it byte-for-byte).
        //
        // ⚠ But short alone is not sufficient: measured, es/fr/ca/tr all produced noise at 2.1-3.4 s in
        // AUTO mode, and all four came out clean at the same lengths once given a reference voice —
        // as did every other language tried that way. Without a reference the loop has to invent
        // speaker and content together from an all-mask start, and that is what fails on short spans.
        // So warn only when both hold, and name the cheaper fix first.
        else if (target < ShortInputTokens && refCodes is null)
            Console.Error.WriteLine($"WARNING: estimated {target} tokens (~{target / 25.0:F1}s) with no reference "
                + $"voice. Under ~{ShortInputTokens / 25}s this model can emit noise rather than speech — its "
                + "corpus is 99% longer than that (median 12s). Pass --voice (with --ref-text) to anchor the "
                + "speaker, or lengthen the text; both fix it. Re-running does not — generation is deterministic.");

        // language: null is the IPA fine-tune's conditioning — see the file header.
        sw.Restart();
        var tokens = tts.GenerateTokens(ipaText, target, condRefText, refCodes, lang: null, instruct: null,
            new OmniVoiceTts.GenConfig(NumStep: numStep));
        float[] audio = tts.DecodeTokens(tokens);
        sw.Stop();
        if (verbose)
            Console.WriteLine($"diffusion ({numStep} steps): transformer {tts.LastTransformerMs:f0} ms, "
                + $"host {tts.LastHostMs:f0} ms");

        // Output post-processing, in the order Python's _post_process_audio uses: remove silence →
        // volume → fade-in/out + zero-pad. Volume: with a reference, un-boost a reference that
        // EncodeReference boosted for being quiet; without one, peak-normalise to 0.5.
        audio = OmniVoiceAudioPost.RemoveSilence(audio, OmniVoiceTts.SampleRate,
            midSilMs: 500, leadSilMs: 100, trailSilMs: 100);
        if (refRms is float rr) { if (rr < 0.1f) Scale(audio, rr / 0.1f); }
        else Normalize(audio, 0.5f);
        audio = OmniVoiceAudioPost.FadeAndPad(audio, OmniVoiceTts.SampleRate);

        var fmt = WaveFormat.CreateIeeeFloatWaveFormat(OmniVoiceTts.SampleRate, 1);
        using (var w = new WaveFileWriter(outPath, fmt)) w.WriteSamples(audio, 0, audio.Length);

        double sec = audio.Length / (double)OmniVoiceTts.SampleRate, ms = sw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"DONE: {sec:F1}s audio in {ms / 1000:F1}s "
            + $"({(ms > 0 ? sec / (ms / 1000) : 0):F1}x real-time) → {outPath}");
        return 0;
    }

    private static string? FirstExisting(params string?[] paths) =>
        paths.FirstOrDefault(p => p is not null && File.Exists(p));

    private static float[] Load24kMono(string path)
    {
        var (samples, sr, ch) = AudioUtils.ReadAudio(path);
        float[] mono = ch > 1 ? AudioUtils.DownmixToMono(samples, ch) : samples;
        return AudioUtils.ResampleMono(mono, sr, OmniVoiceTts.SampleRate);
    }

    private static void Normalize(float[] x, float peak)
    {
        float max = 0;
        foreach (var v in x) max = Math.Max(max, Math.Abs(v));
        if (max < 1e-6f) return;
        float g = peak / max;
        for (int i = 0; i < x.Length; i++) x[i] *= g;
    }

    private static void Scale(float[] x, float g)
    {
        for (int i = 0; i < x.Length; i++) x[i] *= g;
    }

    private static float Rms(float[] x)
    {
        double s = 0;
        foreach (var v in x) s += (double)v * v;
        return x.Length == 0 ? 0f : (float)Math.Sqrt(s / x.Length);
    }

    private static void PrintUsage() => Console.WriteLine("""
        vernacula-tts — IPA-native text-to-speech (vernacula-phonemizer → OmniVoice IPA fine-tune).

        Usage:
          vernacula-tts --lang <code> (--text "..." | --text-file <path>) --out <wav> [options]

        Text:
          --text <str>            Text to speak (spoken verbatim).
          --text-file <path>      Read the text from a file; .md/.markdown is stripped of markup.
          --lang <code>           PHONEMIZER language (en, cy, cmn, zu, …; 192 routed codes). This
                                  never reaches the model — the fine-tune is language-agnostic and
                                  reads the IPA alone.
          --ipa                   --text (and --ref-text) are already IPA; skip the phonemizer.
                                  Mutually exclusive with --lang.
          --print-ipa             Print the IPA that will be fed to the model.
          --data-dir <path>       vernacula-phonemizer data/ tree. Defaults to the submodule at
                                  external/vernacula-phonemizer/data.

        Voice cloning:
          --voice <wav>           Reference clip; any sample rate, downmixed and resampled to 24k.
          --ref-text <str>        Its transcript. Required with --voice.
          --ref-lang <code>       Phonemizer language for --ref-text (defaults to --lang).

        Model:
          --onnx-dir <dir>        Holds omnivoice_transformer.onnx (+ .onnx.data), higgs_encoder.onnx,
                                  higgs_decoder.onnx. Env: OMNIVOICE_ONNX_DIR.
          --model-dir <dir>       k2-fsa-OmniVoice snapshot, for tokenizer.json. Env: OMNIVOICE_MODEL_DIR.
          --tokenizer-json <path> Qwen3 tokenizer.json, if it is somewhere else.
          --diff <path>           IPA fine-tune diff, folded onto the base transformer at load.
                                  Defaults to ipa_diff_v6.onnx in --onnx-dir.
          --no-diff               Run the stock orthographic model. IPA input will NOT be read as
                                  phonemes; this is for A/B comparison, not for synthesis.
          --transformer-file <p>  The transformer to use; an absolute path is honoured. With --diff
                                  this MUST be the base whose weights match the k2-fsa/OmniVoice
                                  checkpoint — the diff carries no fingerprint, so a mismatched base
                                  yields a quietly wrong model rather than an error.

        Generation:
          --out <wav>             Output path (24 kHz mono float32 WAV). Required.
          --num-step <n>          Diffusion steps (default 32).
          --target-tokens <n>     Override the duration estimate (25 tokens ≈ 1 s).
          --ep cpu|cuda|auto      Execution provider (default auto). CUDA runs full fp32; TF32 is
                                  disabled because the diffusion loop degrades into noise under it.
                                  The IPA diff works on every provider (it is a graph rewrite, not a
                                  weight fold), so CUDA needs no pre-merged model.
          --verbose, -v           Per-graph load times, the IPA, and diffusion timings.

        Examples:
          vernacula-tts --lang en --text "Hello world." --out hello.wav
          vernacula-tts --lang cy --text-file page.md --voice ref.wav --ref-text "..." --out out.wav
          vernacula-tts --ipa --text "hɛlˈoʊ wˈɜːld." --print-ipa --out out.wav
        """);
}
