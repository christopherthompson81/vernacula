using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Channels;
using System.Threading.Tasks;
using Avalonia.Threading;
// NAudio.Wave SPANS TWO PACKAGES IN NAudio 3. BufferedWaveProvider, AudioFileReader and
// WaveFormat live in NAudio.Core and resolve on every target framework; WaveOut lives
// in NAudio.WinMM, which ships net9.0-windows7.0 only. Hence the net10.0;net10.0-windows
// multi-target on this project, and `#if WINDOWS` rather than OperatingSystem.IsWindows()
// wherever WaveOut is touched - see Vernacula.Avalonia.csproj.
using NAudio.Wave;

namespace Vernacula.App.Services.Tts;

/// <summary>
/// Cross-platform streaming audio playback for the reader UI. Mirrors
/// the platform split used by Vernacula.Avalonia.TranscriptEditorViewModel
/// (WaveOut on Windows, ffplay elsewhere) but in streaming form:
///
///   Windows: NAudio's <see cref="BufferedWaveProvider"/> fed into
///            <see cref="WaveOut"/>. Samples are appended as
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
#if WINDOWS
    private BufferedWaveProvider? _bufferedProvider;
    // WaveOut, not WaveOutEvent: NAudio 3 renamed the event-callback device to WaveOut
    // (NAudio 2's message-loop WaveOut became WaveOutWindow). WaveOutEvent survives as an
    // obsolete alias, so this is a rename with no behaviour change.
    private WaveOut? _waveOut;
    // Held alongside _waveOut on the file-playback path so we can dispose
    // it in Stop(). WaveOut.Dispose() does NOT dispose the WaveStream
    // it was given via Init — without holding our own ref, every Play
    // would leak an open file handle (Windows keeps the WAV locked too).
    private AudioFileReader? _fileReader;
#endif
    private Process? _ffplayProcess;
    private Stream? _ffplayStdin;

    // Audio is enqueued via AppendSamples and drained by a background
    // writer task. Without this decoupling, AppendSamples would block
    // its caller (the alignment chain) under ffplay-stdin or
    // BufferedWaveProvider backpressure, stalling subsequent chunks
    // until the audio backend caught up — chunks would land in bursts
    // instead of streaming.
    private Channel<float[]>? _audioChannel;
    private Task? _writerTask;

    private DispatcherTimer? _tickTimer;
    private DateTime _startedUtc;
    private int _sampleRate;
    private bool _endOfStream;

    private readonly object _totalLock = new();
    private double _totalEstimatedSec;

    private bool _isPlaying;
    public bool IsPlaying
    {
        get => _isPlaying;
        private set
        {
            if (_isPlaying == value) return;
            _isPlaying = value;
            IsPlayingChanged?.Invoke(value);
        }
    }

    private bool _isPaused;
    public bool IsPaused
    {
        get => _isPaused;
        private set
        {
            if (_isPaused == value) return;
            _isPaused = value;
            IsPausedChanged?.Invoke(value);
        }
    }

    public double PositionSeconds { get; private set; }
    private DateTime _pausedAt;

    /// <summary>Running best-known total duration of the audio that's
    /// been queued. Grows as chunks append during streaming; matches the
    /// final audio duration after <see cref="EndOfStream"/>. Thread-safe.</summary>
    public double TotalEstimatedSeconds
    {
        get { lock (_totalLock) return _totalEstimatedSec; }
    }

    public event Action<double>? PositionChanged;
    public event Action<double>? PlaybackStopped;

    /// <summary>Fired whenever <see cref="TotalEstimatedSeconds"/>
    /// changes — i.e. on every <see cref="AppendSamples"/> call and when
    /// <see cref="SeekIntoFile"/> sets a known total. Lets UI refresh
    /// the position label even when the tick timer isn't running (e.g.
    /// after the user clicked Stop while synth is still emitting).</summary>
    public event Action<double>? TotalChanged;

    /// <summary>Fired when <see cref="IsPlaying"/> transitions. Lets the
    /// UI re-query the Stop/Play CanExecute predicates.</summary>
    public event Action<bool>? IsPlayingChanged;

    /// <summary>Fired when <see cref="IsPaused"/> transitions. Lets the
    /// UI swap the Play/Pause/Resume button label.</summary>
    public event Action<bool>? IsPausedChanged;

#if WINDOWS
    public bool CanPlayOnThisPlatform => true;
#else
    // NOT OperatingSystem.IsWindows(). The net10.0 build carries no WinMM, so on a Windows
    // host running it the WaveOut path does not exist and ffplay is the only backend; asking
    // the OS would claim playback works and then take a branch that was never compiled.
    public bool CanPlayOnThisPlatform => FfplayPath is not null;
#endif
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
        lock (_totalLock) _totalEstimatedSec = 0;
        PositionSeconds = 0;

#if WINDOWS
        var fmt = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1);
        // 60 s of headroom — the writer task throttles when the
        // buffer is more than half-full, so this is the absolute
        // ceiling rather than the working size.
        //
        // NAudio 3 made BufferLength read-only and moved sizing into the constructor as a
        // duration, so the byte arithmetic that used to spell this out is gone; the format
        // already knows its own byte rate.
        _bufferedProvider = new BufferedWaveProvider(fmt, TimeSpan.FromSeconds(60))
        {
            DiscardOnBufferOverflow = false,
        };
        _waveOut = new WaveOut();
        _waveOut.Init(_bufferedProvider);
        _waveOut.Play();
#else
        if (FfplayPath is null)
            throw new InvalidOperationException(UnavailableReason!);
        StartFfplayStdin(sampleRate);
#endif

        // Unbounded so AppendSamples NEVER blocks its caller — the synth
        // pipeline must stay free to produce the next chunk while audio
        // backpressure naturally throttles the writer (not the producer).
        _audioChannel = Channel.CreateUnbounded<float[]>(new UnboundedChannelOptions
        {
            SingleReader = true,
            SingleWriter = false,
        });
        _writerTask = Task.Run(WriteLoop);

        IsPlaying = true;
        _startedUtc = DateTime.UtcNow;
        StartTickTimer();
    }

    /// <summary>Append a chunk of float32 mono samples. Non-blocking:
    /// enqueues to an internal channel that a background writer task
    /// drains into the audio backend. Updating the total here (rather
    /// than in the writer) keeps <see cref="TotalEstimatedSeconds"/>
    /// accurate to what's been *produced*, not what's been played.</summary>
    public void AppendSamples(float[] samples)
    {
        if (samples.Length == 0) return;
        double newTotal;
        lock (_totalLock)
        {
            _totalEstimatedSec += samples.Length / (double)_sampleRate;
            newTotal = _totalEstimatedSec;
        }
        TotalChanged?.Invoke(newTotal);
        _audioChannel?.Writer.TryWrite(samples);
    }

    /// <summary>Background drain loop. Reads from the audio channel and
    /// writes to whichever backend is active. On Windows it throttles
    /// against the BufferedWaveProvider so we don't blow the 60 s cap;
    /// on ffplay the stdin <see cref="Stream.Write"/> blocks naturally
    /// when the pipe is full (kernel-level backpressure).</summary>
    private async Task WriteLoop()
    {
        var channel = _audioChannel;
        if (channel is null) return;
        try
        {
            await foreach (var samples in channel.Reader.ReadAllAsync())
            {
                var bytes = new byte[samples.Length * 4];
                Buffer.BlockCopy(samples, 0, bytes, 0, bytes.Length);

#if WINDOWS
                {
                    var bp = _bufferedProvider;
                    if (bp is null) break;
                    // Feed the buffer in bounded slices, throttling before each so the
                    // buffer never has to take more than it has room for. A chunk is now a
                    // whole paragraph — tens of seconds is ordinary — and one AddSamples of
                    // that onto a half-full 60 s buffer threw "Buffer full" (which ended the
                    // writer loop and silently dropped every later chunk).
                    int sliceBytes = _sampleRate * 4 * 5;   // 5 s
                    bool tornDown = false;
                    for (int off = 0; off < bytes.Length && !tornDown; off += sliceBytes)
                    {
                        while (bp.BufferedDuration.TotalSeconds > 30)
                        {
                            await Task.Delay(100).ConfigureAwait(false);
                            if (_bufferedProvider is null) return;
                        }
                        int n = Math.Min(sliceBytes, bytes.Length - off);
                        try { bp.AddSamples(bytes, off, n); }
                        catch (InvalidOperationException) { tornDown = true; }
                    }
                    if (tornDown) break;
                }
#else
                {
                    var stdin = _ffplayStdin;
                    if (stdin is null) break;
                    try
                    {
                        stdin.Write(bytes, 0, bytes.Length);
                        stdin.Flush();
                    }
                    catch
                    {
                        // Pipe broken (ffplay exited or killed). Exit the
                        // loop — the tick timer's EOF detection will
                        // notice and Stop() the rest.
                        break;
                    }
                }
#endif
            }
        }
        catch (ChannelClosedException) { /* channel completed by Stop/EndOfStream */ }
    }

    /// <summary>Signal that no more samples will be appended. Completes
    /// the audio channel so the writer task drains and exits; the tick
    /// timer fires <see cref="PlaybackStopped"/> once the buffer empties
    /// (Windows) or ffplay exits (other).</summary>
    public void EndOfStream()
    {
        _endOfStream = true;
        _audioChannel?.Writer.TryComplete();
        // ffplay stdin must be closed AFTER the writer task drains; we
        // can't close it here or any pending WriteLoop iteration would
        // hit a broken pipe. The WriteLoop's `await foreach` exits
        // naturally on channel completion, then EndOfStreamDrain closes
        // stdin to tell ffplay "play out the buffer and exit".
        _ = EndOfStreamDrainAsync();
    }

    private async Task EndOfStreamDrainAsync()
    {
        var writer = _writerTask;
        if (writer is not null)
            try { await writer.ConfigureAwait(false); } catch { }
#if !WINDOWS
        if (_ffplayStdin is not null)
        {
            try { _ffplayStdin.Close(); } catch { /* already closed */ }
        }
#endif
    }

    /// <summary>Suspend audio output without tearing down the backend.
    /// On Windows uses <see cref="WaveOut.Pause"/>; on Linux sends
    /// SIGSTOP to the ffplay process so the kernel freezes it. The
    /// internal audio buffer and writer task stay alive — Resume picks
    /// up exactly where Pause left off. Wall-clock position is frozen
    /// via _pausedAt; Resume shifts _startedUtc forward by the pause
    /// duration so the highlight doesn't jump ahead.</summary>
    public void Pause()
    {
        if (!IsPlaying || IsPaused) return;
#if WINDOWS
        try { _waveOut?.Pause(); } catch { /* device race */ }
#else
        if (_ffplayProcess is not null)
        {
            SendSignalToProcess(_ffplayProcess, "STOP");
        }
#endif
        _pausedAt = DateTime.UtcNow;
        StopTickTimer();
        // Order matters: clear IsPlaying BEFORE setting IsPaused so any
        // subscriber that re-queries derived state sees both in a
        // consistent (paused, not-playing) configuration.
        IsPlaying = false;
        IsPaused = true;
    }

    /// <summary>Resume playback from <see cref="Pause"/>. Shifts the
    /// wall-clock anchor forward by the pause duration so position
    /// resumes where it stopped, restarts the tick timer, and unfreezes
    /// the audio backend.</summary>
    public void Resume()
    {
        if (!IsPaused) return;
        var pauseDuration = DateTime.UtcNow - _pausedAt;
        _startedUtc = _startedUtc.Add(pauseDuration);
#if WINDOWS
        try { _waveOut?.Play(); } catch { /* device race */ }
#else
        if (_ffplayProcess is not null)
        {
            SendSignalToProcess(_ffplayProcess, "CONT");
        }
#endif
        IsPaused = false;
        IsPlaying = true;
        StartTickTimer();
    }

    /// <summary>Best-effort POSIX signal via the kill(1) utility. Used
    /// for SIGSTOP/SIGCONT to pause/resume ffplay. Shelling out (rather
    /// than P/Invoke libc) avoids hard-coding signal numbers, which
    /// differ between Linux and macOS.</summary>
    private static void SendSignalToProcess(Process proc, string sig)
    {
        if (proc.HasExited) return;
        try
        {
            using var p = Process.Start(new ProcessStartInfo("kill")
            {
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                ArgumentList = { "-s", sig, proc.Id.ToString() },
            });
            p?.WaitForExit(500);
        }
        catch { /* best effort — pause/resume isn't worth crashing over */ }
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
#if WINDOWS
        // BufferedWaveProvider doesn't expose seek directly — for an
        // in-RAM stream the simplest thing is to truncate the buffer
        // (drop anything before the seek point) and re-anchor the
        // wall-clock. Forward seek = drop bytes; backward seek
        // beyond what's in the buffer = clamp to current position.
        // For MVP, we just re-anchor the wall-clock and let WaveOut
        // continue playing what's in the buffer — meaning seek only
        // affects the HIGHLIGHT, not the audio. Acceptable for
        // click-to-highlight UX; full audio-seek is follow-up.
        if (_bufferedProvider is null) return;
#endif
        // On ffplay the same MVP compromise applies with no buffer to guard on:
        // re-anchor the wall-clock, the highlight jumps, and audio continues from
        // wherever ffplay was. Seeking ffplay needs either a seekable input (file) or
        // restarting the process — both larger surgeries, deferred.
        _startedUtc = DateTime.UtcNow.AddSeconds(-seconds);
        PositionSeconds = seconds;
        PositionChanged?.Invoke(PositionSeconds);
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
        lock (_totalLock) _totalEstimatedSec = audioDurationSec;
        TotalChanged?.Invoke(audioDurationSec);
        seconds = Math.Clamp(seconds, 0, audioDurationSec);
        _endOfStream = true;

#if WINDOWS
        {
            _fileReader = new AudioFileReader(audioPath)
            {
                CurrentTime = TimeSpan.FromSeconds(seconds),
            };
            var player = new WaveOut();
            _waveOut = player;
            _waveOut.Init(_fileReader);
            // ⚠ ONLY STOP IF THIS PLAYER IS STILL THE CURRENT ONE. A seek during playback
            // Stop()s the previous WaveOut, whose PlaybackStopped is delivered
            // asynchronously — after the new player has started — and an unconditional
            // Stop() there tore the new playback down within milliseconds.
            _waveOut.PlaybackStopped += (sender, _) => Dispatcher.UIThread.InvokeAsync(() =>
            {
                if (ReferenceEquals(sender, _waveOut)) Stop();
            });
            _waveOut.Play();
        }
#else
        {
            if (FfplayPath is null)
                throw new InvalidOperationException(UnavailableReason!);
            StartFfplayFile(audioPath, seconds);
        }
#endif

        IsPlaying = true;
        PositionSeconds = seconds;
        _startedUtc = DateTime.UtcNow.AddSeconds(-seconds);
        StartTickTimer();
    }

    public void Stop()
    {
        // Allow Stop after Pause too — without IsPaused in this check
        // we'd early-return because the timer was stopped by Pause and
        // IsPlaying is false during pause.
#if WINDOWS
        if (!IsPlaying && !IsPaused && _waveOut is null && _ffplayProcess is null) return;
#else
        if (!IsPlaying && !IsPaused && _ffplayProcess is null) return;
#endif
        StopTickTimer();

        // ORDER MATTERS for immediate-stop UX. Audio backends are torn
        // down FIRST so output goes silent right now. The channel +
        // writer-task cleanup happens after — those don't produce
        // sound, just internal state.
        //
        // Old order closed ffplay stdin before kill, which let ffplay
        // drain its internal decoded-PCM buffer (sometimes seconds) on
        // -autoexit before the kill landed. Killing first prevents
        // that drain.
#if WINDOWS
        if (_waveOut is not null)
        {
            try { _waveOut.Stop(); } catch { /* shutdown race */ }
            _waveOut.Dispose();
            _waveOut = null;
        }
        if (_fileReader is not null)
        {
            _fileReader.Dispose();
            _fileReader = null;
        }
#endif
        if (_ffplayProcess is not null)
        {
            try { if (!_ffplayProcess.HasExited) _ffplayProcess.Kill(entireProcessTree: true); }
            catch { /* shutdown race */ }
            _ffplayProcess.Dispose();
            _ffplayProcess = null;
        }
        if (_ffplayStdin is not null)
        {
            try { _ffplayStdin.Close(); } catch { }
            _ffplayStdin = null;
        }
        // Channel cleanup last; writer task will hit a broken pipe /
        // disposed buffer and exit its loop on its own.
        _audioChannel?.Writer.TryComplete();
        _audioChannel = null;
        _writerTask = null;
#if WINDOWS
        _bufferedProvider = null;
#endif

        var finalPos = PositionSeconds;
        IsPaused = false;
        IsPlaying = false;
        PlaybackStopped?.Invoke(finalPos);
    }

    private void StartTickTimer()
    {
        _tickTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(50) };
        _tickTimer.Tick += (_, _) =>
        {
            double total = TotalEstimatedSeconds;
            double pos = (DateTime.UtcNow - _startedUtc).TotalSeconds;
            // Clamp to estimated total (grows as chunks arrive). If we
            // outrun the buffer (clock past the last appended sample),
            // freeze at the estimate until more arrives — better than
            // showing fake forward motion.
            if (total > 0 && pos > total) pos = total;
            PositionSeconds = pos;
            PositionChanged?.Invoke(PositionSeconds);

            // Natural EOF: position reached the estimate AND no more
            // samples are coming. (Without _endOfStream we'd stop early
            // every time the user gets ahead of synthesis.)
            if (_endOfStream && total > 0 && PositionSeconds >= total)
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
        // Same guard as the WaveOut handler: a seek replaces the process, and the old one's
        // Exited must not stop the new one.
        p.Exited += (sender, _) => Dispatcher.UIThread.InvokeAsync(() =>
        {
            if (ReferenceEquals(sender, _ffplayProcess)) Stop();
        });
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
