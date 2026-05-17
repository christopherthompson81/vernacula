using System;
using System.Diagnostics;
using System.IO;
using Avalonia.Threading;
using NAudio.Wave;

namespace Chatterbox.App.Services;

/// <summary>
/// Cross-platform streaming audio playback for the reader UI. Mirrors
/// the platform split used by Vernacula.Avalonia.TranscriptEditorViewModel
/// (WaveOutEvent on Windows, ffplay elsewhere) but in streaming form:
///
///   Windows: NAudio's <see cref="BufferedWaveProvider"/> fed into
///            <see cref="WaveOutEvent"/>. Samples are appended as
///            synthesis produces them; WaveOut pulls from the buffer
///            continuously.
///   Other  : <c>ffplay -f f32le -ar 24000 -ac 1 -i pipe:0</c>. Raw
///            float32 PCM is written to ffplay's stdin chunk-by-chunk;
///            it plays continuously and exits on EOF.
///
/// A <see cref="DispatcherTimer"/> ticks at 50 ms regardless of
/// backend, raising <see cref="PositionChanged"/> with the current
/// playback time in seconds — the word-highlight loop binds to that
/// event.
///
/// Lifecycle: <see cref="StartStreaming"/> opens the playback pipeline
/// for a given sample format; <see cref="AppendSamples"/> pushes audio
/// (returns when the bytes are queued, not when they've played);
/// <see cref="EndOfStream"/> signals no more samples will come (so
/// <see cref="PlaybackStopped"/> can fire when the buffer drains);
/// <see cref="Stop"/> tears down immediately. <see cref="SeekTo"/>
/// jumps to an arbitrary position (within what's been buffered);
/// behavior past-EOB is implementation-defined.
/// </summary>
public sealed class PlaybackService : IDisposable
{
    private static readonly string? FfplayPath = FindExecutable("ffplay");

    // Backend state (only one set in use at a time).
    private BufferedWaveProvider? _bufferedProvider;
    private WaveOutEvent? _waveOut;
    private Process? _ffplayProcess;
    private Stream? _ffplayStdin;

    private DispatcherTimer? _tickTimer;
    private DateTime _startedUtc;
    private int _sampleRate;
    private bool _endOfStream;
    private double _totalEstimatedSec;   // best-known total duration (updates as chunks append)

    public bool IsPlaying { get; private set; }
    public double PositionSeconds { get; private set; }

    public event Action<double>? PositionChanged;
    public event Action<double>? PlaybackStopped;

    public bool CanPlayOnThisPlatform => OperatingSystem.IsWindows() || FfplayPath is not null;
    public string? UnavailableReason => CanPlayOnThisPlatform
        ? null
        : "Audio playback requires Windows audio output or an `ffplay` executable in PATH.";

    /// <summary>Open the playback pipeline for streaming and start it
    /// playing immediately (silence until samples arrive). Calls
    /// <see cref="Stop"/> on any in-progress playback first.</summary>
    public void StartStreaming(int sampleRate, int channels)
    {
        if (channels != 1)
            throw new NotSupportedException("MVP supports mono only — Chatterbox always outputs mono.");
        Stop();

        _sampleRate = sampleRate;
        _endOfStream = false;
        _totalEstimatedSec = 0;
        PositionSeconds = 0;

        if (OperatingSystem.IsWindows())
        {
            var fmt = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1);
            // 30 s of headroom — long-form chunks deliver in 1-5 s
            // bursts, vocoder is faster than realtime, so this never
            // backs up in practice but the cap protects against runaway
            // memory in pathological cases.
            _bufferedProvider = new BufferedWaveProvider(fmt)
            {
                BufferLength = sampleRate * 4 /* bytes per float */ * 30,
                DiscardOnBufferOverflow = false,
            };
            _waveOut = new WaveOutEvent();
            _waveOut.Init(_bufferedProvider);
            _waveOut.Play();
        }
        else
        {
            if (FfplayPath is null)
                throw new InvalidOperationException(UnavailableReason!);
            StartFfplayStdin(sampleRate);
        }

        IsPlaying = true;
        _startedUtc = DateTime.UtcNow;
        StartTickTimer();
    }

    /// <summary>Append a chunk of float32 mono samples (Chatterbox's
    /// native format). Safe to call from any thread. Returns when the
    /// bytes are queued, not when they've played.</summary>
    public void AppendSamples(float[] samples)
    {
        if (samples.Length == 0) return;
        _totalEstimatedSec += samples.Length / (double)_sampleRate;

        if (OperatingSystem.IsWindows())
        {
            if (_bufferedProvider is null) return;
            // BufferedWaveProvider takes a byte[] (it's PCM-agnostic).
            // We're feeding IEEE-754 float32, so 4 bytes per sample.
            var bytes = new byte[samples.Length * 4];
            Buffer.BlockCopy(samples, 0, bytes, 0, bytes.Length);
            _bufferedProvider.AddSamples(bytes, 0, bytes.Length);
        }
        else
        {
            if (_ffplayStdin is null) return;
            var bytes = new byte[samples.Length * 4];
            Buffer.BlockCopy(samples, 0, bytes, 0, bytes.Length);
            try
            {
                _ffplayStdin.Write(bytes, 0, bytes.Length);
                _ffplayStdin.Flush();
            }
            catch (Exception)
            {
                // Pipe broken (ffplay exited or was killed). The tick
                // timer's EOF detection will pick it up on the next tick.
            }
        }
    }

    /// <summary>Signal that no more samples will be appended. The tick
    /// timer will fire <see cref="PlaybackStopped"/> once the buffer
    /// drains (Windows) or ffplay exits (other).</summary>
    public void EndOfStream()
    {
        _endOfStream = true;
        if (!OperatingSystem.IsWindows() && _ffplayStdin is not null)
        {
            // Closing stdin tells ffplay "no more input"; it'll play to
            // the end of its decoded buffer then exit naturally.
            try { _ffplayStdin.Close(); } catch { /* already closed */ }
        }
    }

    /// <summary>Jump to an arbitrary position. On Windows this works on
    /// whatever's been buffered. On ffplay it restarts the player at the
    /// new offset, which only works when streaming has finished (we'd
    /// need a full WAV on disk to seek into). MVP: seek before
    /// EndOfStream is ignored on the ffplay path with a console note;
    /// post-EOF seek works because the caller can pass the on-disk WAV
    /// via <see cref="SeekIntoFile"/> instead.</summary>
    public void SeekTo(double seconds)
    {
        if (OperatingSystem.IsWindows())
        {
            if (_bufferedProvider is null) return;
            // BufferedWaveProvider doesn't expose seek directly — for an
            // in-RAM stream the simplest thing is to truncate the buffer
            // (drop anything before the seek point) and re-anchor the
            // wall-clock. Forward seek = drop bytes; backward seek
            // beyond what's in the buffer = clamp to current position.
            // For MVP, we just re-anchor the wall-clock and let WaveOut
            // continue playing what's in the buffer — meaning seek only
            // affects the HIGHLIGHT, not the audio. Acceptable for
            // click-to-highlight UX; full audio-seek is follow-up.
            _startedUtc = DateTime.UtcNow.AddSeconds(-seconds);
            PositionSeconds = seconds;
            PositionChanged?.Invoke(PositionSeconds);
        }
        else
        {
            // Same MVP compromise: re-anchor wall-clock, highlight jumps,
            // audio continues from wherever ffplay was. Seeking ffplay
            // requires either a seekable input (file) or restarting the
            // process — both are larger surgeries deferred for now.
            _startedUtc = DateTime.UtcNow.AddSeconds(-seconds);
            PositionSeconds = seconds;
            PositionChanged?.Invoke(PositionSeconds);
        }
    }

    /// <summary>Post-streaming seek into the on-disk WAV file. Restarts
    /// the playback pipeline reading from the file (instead of the
    /// stream) and jumps to <paramref name="seconds"/>. Use this for
    /// scrubbing a finished synthesis where you have the full WAV.</summary>
    public void SeekIntoFile(string audioPath, double audioDurationSec, double seconds)
    {
        Stop();
        if (!File.Exists(audioPath))
            throw new FileNotFoundException("Audio file not found.", audioPath);

        _sampleRate = 0;  // unused on the file-playback path
        _totalEstimatedSec = audioDurationSec;
        seconds = Math.Clamp(seconds, 0, audioDurationSec);
        _endOfStream = true;

        if (OperatingSystem.IsWindows())
        {
            var reader = new AudioFileReader(audioPath)
            {
                CurrentTime = TimeSpan.FromSeconds(seconds),
            };
            _waveOut = new WaveOutEvent();
            _waveOut.Init(reader);
            _waveOut.PlaybackStopped += (_, _) => Dispatcher.UIThread.InvokeAsync(Stop);
            _waveOut.Play();
        }
        else
        {
            if (FfplayPath is null)
                throw new InvalidOperationException(UnavailableReason!);
            StartFfplayFile(audioPath, seconds);
        }

        IsPlaying = true;
        PositionSeconds = seconds;
        _startedUtc = DateTime.UtcNow.AddSeconds(-seconds);
        StartTickTimer();
    }

    public void Stop()
    {
        if (!IsPlaying && _tickTimer is null) return;
        StopTickTimer();
        if (_waveOut is not null)
        {
            try { _waveOut.Stop(); } catch { /* shutdown race */ }
            _waveOut.Dispose();
            _waveOut = null;
        }
        _bufferedProvider = null;
        if (_ffplayStdin is not null)
        {
            try { _ffplayStdin.Close(); } catch { }
            _ffplayStdin = null;
        }
        if (_ffplayProcess is not null)
        {
            try { if (!_ffplayProcess.HasExited) _ffplayProcess.Kill(entireProcessTree: true); }
            catch { /* shutdown race */ }
            _ffplayProcess.Dispose();
            _ffplayProcess = null;
        }
        var finalPos = PositionSeconds;
        IsPlaying = false;
        PlaybackStopped?.Invoke(finalPos);
    }

    private void StartTickTimer()
    {
        _tickTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(50) };
        _tickTimer.Tick += (_, _) =>
        {
            double pos = (DateTime.UtcNow - _startedUtc).TotalSeconds;
            // Clamp to estimated total (grows as chunks arrive). If we
            // outrun the buffer (clock past the last appended sample),
            // freeze at the estimate until more arrives — better than
            // showing fake forward motion.
            if (_totalEstimatedSec > 0 && pos > _totalEstimatedSec)
                pos = _totalEstimatedSec;
            PositionSeconds = pos;
            PositionChanged?.Invoke(PositionSeconds);

            // Natural EOF: position reached the estimate AND no more
            // samples are coming. (Without _endOfStream we'd stop early
            // every time the user gets ahead of synthesis.)
            if (_endOfStream && _totalEstimatedSec > 0 && PositionSeconds >= _totalEstimatedSec)
                Stop();
        };
        _tickTimer.Start();
    }

    private void StopTickTimer() { _tickTimer?.Stop(); _tickTimer = null; }

    private void StartFfplayStdin(int sampleRate)
    {
        var psi = MakeFfplayBasePsi();
        psi.ArgumentList.Add("-f"); psi.ArgumentList.Add("f32le");
        psi.ArgumentList.Add("-ar"); psi.ArgumentList.Add(sampleRate.ToString());
        psi.ArgumentList.Add("-ac"); psi.ArgumentList.Add("1");
        psi.ArgumentList.Add("-i");  psi.ArgumentList.Add("pipe:0");
        psi.RedirectStandardInput = true;
        StartFfplay(psi);
    }

    private void StartFfplayFile(string audioPath, double startSec)
    {
        var psi = MakeFfplayBasePsi();
        psi.ArgumentList.Add("-ss"); psi.ArgumentList.Add(startSec.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture));
        psi.ArgumentList.Add("-i");  psi.ArgumentList.Add(audioPath);
        StartFfplay(psi);
    }

    private ProcessStartInfo MakeFfplayBasePsi()
    {
        var psi = new ProcessStartInfo(FfplayPath!)
        {
            UseShellExecute = false,
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            CreateNoWindow = true,
        };
        psi.ArgumentList.Add("-nodisp");
        psi.ArgumentList.Add("-autoexit");
        psi.ArgumentList.Add("-loglevel"); psi.ArgumentList.Add("error");
        return psi;
    }

    private void StartFfplay(ProcessStartInfo psi)
    {
        var p = new Process { StartInfo = psi, EnableRaisingEvents = true };
        p.Exited += (_, _) => Dispatcher.UIThread.InvokeAsync(Stop);
        if (!p.Start())
        {
            p.Dispose();
            throw new InvalidOperationException("Failed to start ffplay.");
        }
        _ = p.StandardOutput.ReadToEndAsync();
        _ = p.StandardError.ReadToEndAsync();
        _ffplayProcess = p;
        if (psi.RedirectStandardInput) _ffplayStdin = p.StandardInput.BaseStream;
    }

    private static string? FindExecutable(string fileName)
    {
        var pathEnv = Environment.GetEnvironmentVariable("PATH");
        if (string.IsNullOrEmpty(pathEnv)) return null;
        char sep = OperatingSystem.IsWindows() ? ';' : ':';
        string[] exts = OperatingSystem.IsWindows() ? new[] { ".exe", ".bat", ".cmd" } : new[] { "" };
        foreach (var dir in pathEnv.Split(sep))
        {
            if (string.IsNullOrWhiteSpace(dir)) continue;
            foreach (var ext in exts)
            {
                var candidate = Path.Combine(dir, fileName + ext);
                if (File.Exists(candidate)) return candidate;
            }
        }
        return null;
    }

    public void Dispose() => Stop();
}
