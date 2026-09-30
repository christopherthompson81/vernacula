// Nemotron-3-Diarization port parity harness (#246).
//
// Runs SortformerStreamer over WAV files and writes, per file and per model:
//
//   <stem>.<model>.preds.f32   raw reported probabilities, float32 little-endian [T, S]
//   <stem>.<model>.rttm        the segments the application would emit
//
// scripts/nemo_export/nemotron3_diarization_fidelity.py compares the Nemotron-3 preds against
// NeMo's own forward_streaming and scores both models' RTTMs against reference labels.
//
// MANUAL TOOL -- not in Vernacula.slnx and not run by CI, like tests/SortformerRouteCheck.
//
// Run:  dotnet run --project tests/Nemotron3DiarizationParity -p:EP=Cpu -- \
//           <models-root> <out-dir> [--v21] a.wav b.wav ...
//
// <models-root> holds nemotron3_diarization/nemotron-3-diarization.onnx (and, with --v21,
// sortformer/diar_streaming_sortformer_4spk-v2.1.onnx).

using System.Diagnostics;
using System.Globalization;
using Vernacula.Base;
using Vernacula.Base.Models;

if (args.Length < 3)
{
    Console.Error.WriteLine("usage: <models-root> <out-dir> [--v21] a.wav [b.wav ...]");
    return 2;
}

string modelsRoot = args[0];
string outDir     = args[1];
bool   alsoV21    = args.Contains("--v21");
var    wavs       = args.Skip(2).Where(a => a != "--v21").ToList();
Directory.CreateDirectory(outDir);

var models = new List<(SortformerModel Model, string Tag)> { (SortformerModel.Nemotron3, "nemotron3") };
if (alsoV21) models.Add((SortformerModel.StreamingSortformerV21, "v21"));

foreach (var (model, tag) in models)
{
    var load = Stopwatch.StartNew();
    using var s = new SortformerStreamer(modelsRoot, ExecutionProvider.Cpu, model);
    Console.WriteLine($"[{tag}] loaded in {load.ElapsedMilliseconds} ms: {s.Profile.NumSpeakers} spk, " +
                      $"chunk {s.Profile.ChunkLength}+{s.Profile.ChunkRightContext} / fifo {s.Profile.FifoLength} / " +
                      $"cache {s.Profile.SpeakerCacheLength}, frame {s.Profile.FrameDuration * 1000:0} ms");

    foreach (string wav in wavs)
    {
        var (samples, sr, ch) = AudioUtils.ReadAudio(wav);
        if (sr != Config.SampleRate || ch != 1)
            throw new InvalidDataException($"{wav}: need 16 kHz mono, got {sr} Hz x {ch}");

        s.ResetState();
        var sw = Stopwatch.StartNew();
        var mel = AudioUtils.LogMelSpectrogram(samples);
        var (totalFrames, stride, numChunks) = s.GetPredParams(mel);
        var all = s.GetPreds(mel, totalFrames, stride, numChunks).Select(x => x.chunkPreds).ToList();
        var (n, filtered) = s.FilterPreds(all, totalFrames);
        var segs = s.BinarizePredToSegments(n, filtered);
        sw.Stop();

        string stem = Path.GetFileNameWithoutExtension(wav);
        int S = s.Profile.NumSpeakers;
        using (var bw = new BinaryWriter(File.Create(Path.Combine(outDir, $"{stem}.{tag}.preds.f32"))))
            foreach (var chunk in all)
                for (int t = 0; t < chunk.GetLength(0); t++)
                    for (int k = 0; k < S; k++)
                        bw.Write(chunk[t, k]);

        using (var tw = new StreamWriter(Path.Combine(outDir, $"{stem}.{tag}.rttm")))
            foreach (var (st, en, spk) in segs)
                tw.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"SPEAKER {stem} 1 {st:0.000} {en - st:0.000} <NA> <NA> {spk} <NA> <NA>"));

        double secs = samples.Length / (double)sr;
        Console.WriteLine($"[{tag}] {stem}: {secs:0.0}s, {numChunks} chunks, {n} frames, {segs.Count} segments, " +
                          $"{segs.Select(x => x.spkId).Distinct().Count()} speakers, " +
                          $"{sw.ElapsedMilliseconds} ms (RTF {sw.Elapsed.TotalSeconds / secs:0.000})");
    }
}
return 0;
