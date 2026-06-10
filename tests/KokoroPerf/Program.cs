// Kokoro TTS performance harness — latency / real-time-factor across providers.
//
//   dotnet run -c Release --project tests/KokoroPerf -- \
//       --onnx-dir <dir> --data-dir <espeak data/> --ep cuda --iters 10
//
// Reports session-load time and steady-state per-utterance latency + RTF
// (audio_seconds / inference_seconds; higher = faster than real time) over a
// few utterance lengths. Doubles as the acceptance benchmark for the later
// fp16/int8 quantization phase.

using System.Diagnostics;
using Chatterbox.Base;
using Vernacula.Base.Models;

string onnxDir = "/home/chris/Programming/vernacula/scripts/kokoro_export/external/kokoro_onnx";
string dataDir = "/home/chris/Programming/vernacula/external/espeak-ng-portable/data";
var ep = ExecutionProvider.Cuda;
var voice = "af_heart";
var iters = 10;

for (var i = 0; i < args.Length - 1; i++)
{
    switch (args[i])
    {
        case "--onnx-dir": onnxDir = args[++i]; break;
        case "--data-dir": dataDir = args[++i]; break;
        case "--voice": voice = args[++i]; break;
        case "--iters": iters = int.Parse(args[++i]); break;
        case "--ep": ep = Enum.Parse<ExecutionProvider>(args[++i], ignoreCase: true); break;
    }
}

(string Label, string Text)[] cases =
[
    ("short ", "Hello there."),
    ("medium", "The quick brown fox jumps over the lazy dog."),
    ("long  ", "In the quiet hours before dawn, the old lighthouse keeper climbed the spiral "
             + "stair, lit the great lamp, and watched its slow beam sweep across the water."),
];

Console.WriteLine($"== Kokoro perf : ep={ep} voice={voice} iters={iters} ==");

long loadMs = 0;
void OnLoad(SessionLoadEvent e) => loadMs = e.ElapsedMs;

var swLoad = Stopwatch.StartNew();
using var tts = new KokoroTts(onnxDir, dataDir, ep, OnLoad);
swLoad.Stop();
Console.WriteLine($"session load: {loadMs} ms (ctor incl. phonemizer+voice setup: {swLoad.ElapsedMilliseconds} ms)\n");

const int sampleRate = 24_000;
Console.WriteLine($"{"case",-7} {"phonemes",8} {"audio_s",8} {"warm_ms",8} {"med_ms",8} {"p90_ms",8} {"RTF",7}");

foreach (var (label, text) in cases)
{
    var ps = tts.ToPhonemes(text);

    // warmup (also the first-call cost, which includes CUDA kernel JIT / cudnn autotune)
    var swWarm = Stopwatch.StartNew();
    var audio = tts.Speak(text, voice);
    swWarm.Stop();

    var times = new List<double>(iters);
    for (var i = 0; i < iters; i++)
    {
        var sw = Stopwatch.StartNew();
        tts.Speak(text, voice);
        sw.Stop();
        times.Add(sw.Elapsed.TotalMilliseconds);
    }
    // dump audio for cross-provider correctness comparison
    var bytes = new byte[audio.Length * sizeof(float)];
    Buffer.BlockCopy(audio, 0, bytes, 0, bytes.Length);
    File.WriteAllBytes($"/tmp/kokoro_{ep}_{label.Trim()}.bin", bytes);

    times.Sort();
    var median = times[times.Count / 2];
    var p90 = times[(int)(times.Count * 0.9)];
    var audioSec = audio.Length / (double)sampleRate;
    var rtf = audioSec / (median / 1000.0);

    Console.WriteLine($"{label,-7} {ps.Length,8} {audioSec,8:F2} {swWarm.ElapsedMilliseconds,8} " +
                      $"{median,8:F1} {p90,8:F1} {rtf,7:F1}x");
}
