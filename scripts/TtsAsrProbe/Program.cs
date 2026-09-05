// TTS / ASR / forced-alignment desync probe.
//
// Investigates whether [UNK] tokens emitted by EnTokenizer cause the
// audio Chatterbox produces to drift from the reference text the NFA
// aligner is given — which would explain the word-highlight desync the
// user reports in the Avalonia reader.
//
// Pipeline:
//   1. Read input markdown → extract → chunk (same as reader app)
//   2. Per chunk: tokenize, dump token IDs + mark UNK positions
//   3. Synthesize the chunk via ChatterboxPipeline
//   4. Run Parakeet ASR on the synthesized audio
//   5. Print a three-column report: input | DiagnosticDecode(tokens) | ASR
//
// Run:
//   dotnet run --project scripts/TtsAsrProbe -c Release -- \
//     --in <markdown>  \
//     --voice <wav>  \
//     --onnx-dir <chatterbox bundle>  \
//     --parakeet-dir <parakeet model dir>  \
//     [--out-wav <wav>] \
//     [--single-chunk]   # only the first chunk (fast)

using Vernacula.Tts.Base;
using Vernacula.Tts.Base.Markdown;
using Vernacula.Tts.Base.Tokenization;
using NAudio.Wave;
using Vernacula.Base;
using Vernacula.Base.Models;

string? inPath = null, voicePath = null, onnxDir = null, parakeetDir = null;
string? nfaDir = null;
string? outWavPath = null;
bool singleChunk = false;

for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--in":           inPath        = args[++i]; break;
        case "--voice":        voicePath     = args[++i]; break;
        case "--onnx-dir":     onnxDir       = args[++i]; break;
        case "--parakeet-dir": parakeetDir   = args[++i]; break;
        case "--nfa-dir":      nfaDir        = args[++i]; break;
        case "--out-wav":      outWavPath    = args[++i]; break;
        case "--single-chunk": singleChunk   = true; break;
        default:
            Console.Error.WriteLine($"Unknown arg: {args[i]}");
            return 2;
    }
}
if (inPath is null || voicePath is null || onnxDir is null || parakeetDir is null)
{
    Console.Error.WriteLine("Required: --in --voice --onnx-dir --parakeet-dir");
    return 2;
}

Console.WriteLine("== TTS / ASR / Alignment Desync Probe ==");
Console.WriteLine($"input:    {inPath}");
Console.WriteLine($"voice:    {voicePath}");
Console.WriteLine($"onnx-dir: {onnxDir}");
Console.WriteLine($"parakeet: {parakeetDir}");
Console.WriteLine();

// ── Phase 1: extract + chunk ─────────────────────────────────────────
var rawText = File.ReadAllText(inPath);
var extracted = MarkdownTextExtractor.Extract(rawText).Text;
var chunks = ParagraphChunker.Chunk(extracted);
Console.WriteLine($"[extract] raw={rawText.Length} chars, extracted={extracted.Length} chars, "
                + $"chunked={chunks.Count} paragraphs");
if (singleChunk && chunks.Count > 1)
{
    Console.WriteLine($"[--single-chunk] truncating to chunk 0 only");
    chunks = new List<string> { chunks[0] };
}

// ── Phase 2: load pipeline + tokenize per chunk ──────────────────────
Console.WriteLine("[load] ChatterboxPipeline...");
using var pipeline = new ChatterboxPipeline(onnxDir, ExecutionProvider.Auto);
if (pipeline.Tokenizer is null)
{
    Console.Error.WriteLine("Pipeline has no tokenizer — pass --tokenizer-json or fix the cache.");
    return 1;
}
var tokenizer = pipeline.Tokenizer;

Console.WriteLine("[load] Parakeet...");
using var parakeet = new Parakeet(parakeetDir);

Vernacula.Base.Alignment.NemoNfaAligner? aligner = null;
if (nfaDir is not null)
{
    Console.WriteLine("[load] NFA aligner...");
    aligner = new Vernacula.Base.Alignment.NemoNfaAligner(nfaDir, ExecutionProvider.Auto);
}

Console.WriteLine("[embed] speaker...");
var spk = pipeline.Embedder.Embed(voicePath);

// ── Phase 3: per-chunk loop ──────────────────────────────────────────
var allChunkWavs = new List<float[]>();
int totalUnk = 0;
for (int ci = 0; ci < chunks.Count; ci++)
{
    string chunkText = chunks[ci];
    Console.WriteLine();
    Console.WriteLine($"================ Chunk {ci + 1}/{chunks.Count} ================");
    Console.WriteLine($"[input ] {Trunc(chunkText)}");

    // Tokenize the bare BPE form (without LM wrappers) so unk counts
    // reflect the text-domain tokenization, not control tokens.
    var bareIds = tokenizer.Encode(chunkText);
    int chunkUnk = bareIds.Count(id => id == tokenizer.UnkToken);
    totalUnk += chunkUnk;
    Console.WriteLine($"[tokens] {bareIds.Length} ids, {chunkUnk} UNK");

    // Highlight unk positions inline against the decoded string.
    string decoded = tokenizer.DiagnosticDecode(bareIds);
    decoded = decoded.Replace("[UNK]", "⟨UNK⟩");
    Console.WriteLine($"[decoded] {Trunc(decoded)}");

    if (chunkUnk > 0)
    {
        // Map UNK token positions back to the source string. Each Rune
        // in the input is one BPE atom before merging — so the nth UNK
        // corresponds to the nth out-of-vocab Rune.
        var unkRunes = new List<string>();
        foreach (var rune in chunkText.EnumerateRunes())
        {
            var s = rune.ToString();
            // Mirror EnTokenizer's vocab check: if there's no single-
            // codepoint entry in vocab, it becomes UNK. We approximate
            // by checking against the [SPACE] substitution + a
            // throwaway encode of just this rune.
            if (s == " ") continue;
            var probe = tokenizer.Encode(s);
            if (probe.Length == 1 && probe[0] == tokenizer.UnkToken)
                unkRunes.Add(s);
            // Two-token results (with one UNK and one merge) also count
            // but for the rough mapping we just need the first.
            else if (probe.Contains(tokenizer.UnkToken))
                unkRunes.Add(s + "?");
        }
        Console.WriteLine($"[unk chars] {string.Join(" ", unkRunes.Take(50).Select(s => $"'{s}'"))}");
    }

    // Synthesize. Use 1024 max LM steps to match Vernacula.Tts.Backends.CLI's
    // override — DefaultMaxLmSteps is only 256, which truncates
    // long-form chunks (the original probe bug + the same bug the
    // Avalonia reader has).
    var wrapped = tokenizer.WrapForLm(chunkText);
    var sw = System.Diagnostics.Stopwatch.StartNew();
    var lmResult = pipeline.Lm.Generate(spk.CondEmb, wrapped, maxSteps: 1024);
    var speechTokens = lmResult.BuildSpeechTokens(spk.AudioTokens);
    var wav24k = pipeline.Vocoder.Synthesize(
        speechTokens, spk.SpeakerEmbeddings, spk.SpeakerFeatures);
    sw.Stop();
    double chunkSec = wav24k.Length / (double)ChatterboxConstants.S3GenSr;
    Console.WriteLine($"[synth ] {chunkSec:F2}s audio in {sw.ElapsedMilliseconds}ms "
                    + $"({lmResult.Steps} LM steps)");
    allChunkWavs.Add(wav24k);

    // ASR this chunk in isolation so we get a per-chunk transcript.
    var wav16k = AudioUtils.AudioTo16000Mono(wav24k, ChatterboxConstants.S3GenSr, channels: 1);
    var segs = new List<(double, double, string)> { (0.0, chunkSec, "spk") };
    string asrText = "";
    foreach (var rec in parakeet.Recognize(segs, wav16k))
        asrText = rec.text;
    Console.WriteLine($"[ASR   ] {Trunc(asrText)}");

    // NFA aligner output as the reader UI sees it. The .Text field of
    // each AlignedWord is what populates the word-grid buttons.
    if (aligner is not null)
    {
        var words = aligner.Align(wav16k, chunkText, "en");
        var words50 = words.Take(50).Select(w => w.Text).ToList();
        Console.WriteLine($"[align ] {words.Count} words: "
            + string.Join(" | ", words50)
            + (words.Count > 50 ? " | …" : ""));
    }
}

// ── Phase 4: concatenated WAV (if requested) ─────────────────────────
if (outWavPath is not null)
{
    int total = allChunkWavs.Sum(c => c.Length);
    var concat = new float[total];
    int off = 0;
    foreach (var c in allChunkWavs)
    {
        Array.Copy(c, 0, concat, off, c.Length);
        off += c.Length;
    }
    var fmt = WaveFormat.CreateIeeeFloatWaveFormat(ChatterboxConstants.S3GenSr, 1);
    using var w = new WaveFileWriter(outWavPath, fmt);
    w.WriteSamples(concat, 0, concat.Length);
    Console.WriteLine();
    Console.WriteLine($"[out   ] wrote {outWavPath} ({total / (double)ChatterboxConstants.S3GenSr:F2}s)");
}

Console.WriteLine();
Console.WriteLine($"== TOTAL: {totalUnk} UNK token(s) across {chunks.Count} chunk(s) ==");
return 0;

static string Trunc(string s, int max = 180)
    => s.Length <= max ? s : s.Substring(0, max) + "…";
