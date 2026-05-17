using System;
using System.Diagnostics;
using System.IO;
using Avalonia.Threading;
using NAudio.Wave;

namespace Chatterbox.App.Services;

/// <summary>
/// Cross-platform audio playback for the reader UI. Mirrors the
/// pattern from Vernacula.Avalonia.TranscriptEditorViewModel:
///
///   - Windows: NAudio's WaveOutEvent (WinMM)
///   - Other  : shell out to ffplay via Process.Start
///
/// A <see cref="DispatcherTimer"/> ticks at 50 ms regardless of
/// playback backend, raising <see cref="PositionChanged"/> with the
/// current playback time in seconds — the word-highlight loop binds
/// to that event.
///
/// MVP only supports play / stop. Pause and seek are deliberately
/// out of scope; the next iteration would add them once the highlight
/// loop is verified visually.
/// </summary>
public sealed class PlaybackService : IDisposable
{
    private static readonly string? FfplayPath = FindExecutable("ffplay");

    private WaveOutEvent? _waveOut;
    private AudioFileReader? _reader;
    private Process? _ffplayProcess;
    private DispatcherTimer? _tickTimer;
    private DateTime _startedUtc;
    private string? _audioPath;
    private double _audioDurationSec;

    /// <summary>True when audio is currently playing (or believed to be —
    /// the ffplay process may have already exited and we haven't
    /// noticed yet).</summary>
    public bool IsPlaying { get; private set; }

    /// <summary>Current playback time in seconds from the start of the file.</summary>
    public double PositionSeconds { get; private set; }

    /// <summary>Fired on every <see cref="DispatcherTimer"/> tick during
    /// playback (50 ms). Argument is current position in seconds.</summary>
    public event Action<double>? PositionChanged;

    /// <summary>Fired when playback finishes naturally (EOF) or is stopped
    /// via <see cref="Stop"/>. Argument is the final position.</summary>
    public event Action<double>? PlaybackStopped;

    public bool CanPlayOnThisPlatform => OperatingSystem.IsWindows() || FfplayPath is not null;

    public string? UnavailableReason => CanPlayOnThisPlatform
        ? null
        : "Audio playback requires Windows audio output or an `ffplay` executable in PATH.";

    public void Play(string audioPath, double audioDurationSec)
    {
        Stop();
        if (!File.Exists(audioPath))
            throw new FileNotFoundException("Audio file not found.", audioPath);

        _audioPath = audioPath;
        _audioDurationSec = audioDurationSec;

        if (OperatingSystem.IsWindows())
        {
            _reader = new AudioFileReader(audioPath);
            _waveOut = new WaveOutEvent();
            _waveOut.Init(_reader);
            _waveOut.PlaybackStopped += OnWaveOutStopped;
            _waveOut.Play();
        }
        else
        {
            if (FfplayPath is null)
                throw new InvalidOperationException(UnavailableReason!);
            if (!StartFfplay(audioPath))
                throw new InvalidOperationException("Failed to start ffplay.");
        }

        IsPlaying = true;
        PositionSeconds = 0;
        _startedUtc = DateTime.UtcNow;
        StartTickTimer();
    }

    public void Stop()
    {
        if (!IsPlaying && _tickTimer is null) return;

        StopTickTimer();
        if (_waveOut is not null)
        {
            _waveOut.PlaybackStopped -= OnWaveOutStopped;
            try { _waveOut.Stop(); } catch { /* shutdown race */ }
            _waveOut.Dispose();
            _waveOut = null;
        }
        if (_reader is not null) { _reader.Dispose(); _reader = null; }
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
            // Wall-clock-derived position. The audio backend's own clock
            // would be slightly more accurate but adds backend-specific
            // plumbing (WaveOut.GetPosition / parse ffplay -progress);
            // wall-clock is good enough for the highlight loop's 50 ms
            // tick + the alignment's per-word boundaries.
            double pos = (DateTime.UtcNow - _startedUtc).TotalSeconds;
            PositionSeconds = Math.Min(pos, _audioDurationSec);
            PositionChanged?.Invoke(PositionSeconds);
            if (PositionSeconds >= _audioDurationSec) Stop();
        };
        _tickTimer.Start();
    }

    private void StopTickTimer()
    {
        _tickTimer?.Stop();
        _tickTimer = null;
    }

    private void OnWaveOutStopped(object? sender, StoppedEventArgs e)
        => Dispatcher.UIThread.InvokeAsync(Stop);

    private bool StartFfplay(string audioPath)
    {
        if (FfplayPath is null) return false;
        var psi = new ProcessStartInfo(FfplayPath)
        {
            UseShellExecute = false,
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            CreateNoWindow = true,
        };
        psi.ArgumentList.Add("-nodisp");
        psi.ArgumentList.Add("-autoexit");
        psi.ArgumentList.Add("-loglevel"); psi.ArgumentList.Add("error");
        psi.ArgumentList.Add(audioPath);

        var p = new Process { StartInfo = psi, EnableRaisingEvents = true };
        p.Exited += (_, _) => Dispatcher.UIThread.InvokeAsync(Stop);
        if (!p.Start())
        {
            p.Dispose();
            return false;
        }
        // Drain stdout/stderr so the pipe doesn't backpressure ffplay.
        _ = p.StandardOutput.ReadToEndAsync();
        _ = p.StandardError.ReadToEndAsync();
        _ffplayProcess = p;
        return true;
    }

    /// <summary>Locate an executable in the user's PATH. Returns null when
    /// not found. Same logic as Vernacula.Avalonia's helper.</summary>
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
