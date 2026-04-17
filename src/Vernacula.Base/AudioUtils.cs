using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace Vernacula.Base;

/// <summary>
/// Core audio signal processing utilities — mel filterbank, STFT, resampling,
/// log-softmax, and NAudio-based audio loading.
/// WPF-specific features (FFmpeg routing, Excel export) live in the WPF project.
/// </summary>
public static class AudioUtils
{
    // ── Mel filterbank (computed once) ───────────────────────────────────────
    public static readonly float[,] MelFilterbank = CreateMelFilterbank();

    // ── SHA-256 checksum ─────────────────────────────────────────────────────

    /// <summary>Mirrors utils.py sha256_checksum().</summary>
    public static string Sha256Checksum(string path)
    {
        using var sha256 = SHA256.Create();
        using var stream = File.OpenRead(path);
        const int blockSize = 65_536;
        var buffer = new byte[blockSize];
        int read;
        while ((read = stream.Read(buffer, 0, blockSize)) > 0)
            sha256.TransformBlock(buffer, 0, read, null, 0);
        sha256.TransformFinalBlock(Array.Empty<byte>(), 0, 0);
        return BitConverter.ToString(sha256.Hash!).Replace("-", "").ToLowerInvariant();
    }

    // ── Slaney mel conversion ────────────────────────────────────────────────

    /// <summary>Mirrors utils.py hz_to_mel_slaney().</summary>
    public static double HzToMelSlaney(double hz)
    {
        if (hz >= Config.MinLogHz)
            return Config.MinLogMel + Math.Log(hz / Config.MinLogHz) / Config.LogStep;
        return (hz - Config.FMin) / Config.FSp;
    }

    /// <summary>Mirrors utils.py mel_to_hz_slaney().</summary>
    public static double MelToHzSlaney(double mel)
    {
        if (mel >= Config.MinLogMel)
            return Config.MinLogHz * Math.Exp(Config.LogStep * (mel - Config.MinLogMel));
        return Config.FMin + Config.FSp * mel;
    }

    // ── Mel filterbank construction ──────────────────────────────────────────

    /// <summary>
    /// Returns a (NMels, FreqBins) matrix.  Mirrors utils.py create_mel_filterbank().
    /// FreqBins = NFft/2 + 1 = 257.
    /// </summary>
    public static float[,] CreateMelFilterbank()
    {
        int freqBins = Config.NFft / 2 + 1;
        var fb = new float[Config.NMels, freqBins];

        var fftFreqs = new double[freqBins];
        for (int k = 0; k < freqBins; k++)
            fftFreqs[k] = (double)k * Config.SampleRate / Config.NFft;

        double fminMel = HzToMelSlaney(0.0);
        double fmaxMel = HzToMelSlaney(Config.SampleRate / 2.0);
        var melF = new double[Config.NMels + 2];
        for (int i = 0; i <= Config.NMels + 1; i++)
        {
            double m = fminMel + (fmaxMel - fminMel) * i / (Config.NMels + 1);
            melF[i] = MelToHzSlaney(m);
        }

        var fdiff = new double[Config.NMels + 1];
        for (int i = 0; i <= Config.NMels; i++)
            fdiff[i] = melF[i + 1] - melF[i];

        for (int i = 0; i < Config.NMels; i++)
            for (int k = 0; k < freqBins; k++)
            {
                double lower = (fftFreqs[k] - melF[i])     / fdiff[i];
                double upper = (melF[i + 2] - fftFreqs[k]) / fdiff[i + 1];
                fb[i, k] = (float)Math.Max(0.0, Math.Min(lower, upper));
            }

        for (int i = 0; i < Config.NMels; i++)
        {
            float enorm = (float)(2.0 / (melF[i + 2] - melF[i]));
            for (int k = 0; k < freqBins; k++)
                fb[i, k] *= enorm;
        }

        return fb;
    }

    // ── Feature extraction ───────────────────────────────────────────────────

    /// <summary>Mirrors utils.py preemphasis() with coef=0.97.</summary>
    public static float[] Preemphasis(float[] signal)
    {
        var out_ = new float[signal.Length];
        out_[0] = signal[0];
        for (int i = 1; i < signal.Length; i++)
            out_[i] = signal[i] - Config.Preemph * signal[i - 1];
        return out_;
    }

    /// <summary>
    /// Magnitude-squared STFT. Mirrors utils.py stft().
    /// Returns shape (FreqBins=257, nFrames).
    /// </summary>
    public static float[,] Stft(float[] signal)
    {
        int nFft      = Config.NFft;
        int winLength = Config.WinLength;
        int hopLength = Config.HopLength;
        int freqBins  = nFft / 2 + 1;

        double[] hann = Window.HannPeriodic(winLength);
        int winOffset = (nFft - winLength) / 2;
        var fftWindow = new double[nFft];
        for (int i = 0; i < winLength; i++)
            fftWindow[winOffset + i] = hann[i];

        int pad        = nFft / 2;
        int paddedLen  = signal.Length + 2 * pad;
        var padded     = new float[paddedLen];
        Array.Copy(signal, 0, padded, pad, signal.Length);

        int nFrames = (paddedLen - nFft) / hopLength + 1;
        var spec    = new float[freqBins, nFrames];

        // Parallelise over frames — each FFT is independent.
        Parallel.For(0, nFrames, i =>
        {
            int start = i * hopLength;
            var frame = new Complex32[nFft];
            for (int j = 0; j < nFft; j++)
                frame[j] = new Complex32((float)(padded[start + j] * fftWindow[j]), 0f);

            Fourier.Forward(frame, FourierOptions.NoScaling);

            for (int k = 0; k < freqBins; k++)
            {
                float re = frame[k].Real;
                float im = frame[k].Imaginary;
                spec[k, i] = re * re + im * im;
            }
        });

        return spec;
    }

    /// <summary>
    /// Full log-mel spectrogram pipeline.  Mirrors utils.py log_mel_spectrogram().
    /// Returns shape (1, T, NMels).
    /// </summary>
    public static float[,,] LogMelSpectrogram(float[] signal)
    {
        float[] y   = Preemphasis(signal);
        float[,] sp = Stft(y);

        int freqBins = sp.GetLength(0);
        int nFrames  = sp.GetLength(1);

        var melSpec = new float[Config.NMels, nFrames];
        // Parallelise over frames — each frame's mel dot-products are independent.
        var poMel = new ParallelOptions { MaxDegreeOfParallelism = Math.Max(2, Environment.ProcessorCount - 1) };
        Parallel.For(0, nFrames, poMel, t =>
        {
            for (int m = 0; m < Config.NMels; m++)
            {
                float sum = 0f;
                for (int k = 0; k < freqBins; k++)
                    sum += MelFilterbank[m, k] * sp[k, t];
                melSpec[m, t] = sum;
            }
        });

        var result = new float[1, nFrames, Config.NMels];
        // Parallelise log transform over frames.
        Parallel.For(0, nFrames, poMel, t =>
        {
            for (int m = 0; m < Config.NMels; m++)
                result[0, t, m] = (float)Math.Log(melSpec[m, t] + Config.LogZeroGuard);
        });

        return result;
    }

    /// <summary>Mirrors utils.py log_softmax() along last axis.</summary>
    public static float[] LogSoftmax(float[] x)
    {
        float max = float.NegativeInfinity;
        foreach (float v in x)
            if (v > max) max = v;

        double sumExp = 0.0;
        foreach (float v in x)
            sumExp += Math.Exp(v - max);
        float logSumExp = (float)(Math.Log(sumExp) + max);

        var result = new float[x.Length];
        for (int i = 0; i < x.Length; i++)
            result[i] = x[i] - logSumExp;
        return result;
    }

    // ── Audio I/O (NAudio only) ──────────────────────────────────────────────

    /// <summary>
    /// Read an audio file via NAudio (WAV, MP3, FLAC, M4A, OGG, AAC).
    /// Returns interleaved float samples in [-1, 1], the sample rate, and channel count.
    /// For video containers or FFmpeg-only formats use the WPF-side ReadAudio overload.
    /// </summary>
    public static (float[] samples, int sampleRate, int channels) ReadAudio(string path)
    {
        if (!OperatingSystem.IsWindows() &&
            string.Equals(Path.GetExtension(path), ".wav", StringComparison.OrdinalIgnoreCase))
        {
            using var wavReader = new WaveFileReader(path);
            ISampleProvider sampleProvider = wavReader.ToSampleProvider();
            int wavSampleRate = sampleProvider.WaveFormat.SampleRate;
            int wavChannels = sampleProvider.WaveFormat.Channels;

            var wavSamples = new List<float>(wavSampleRate * wavChannels * 10);
            var wavBuffer = new float[8192];
            int wavRead;
            while ((wavRead = sampleProvider.Read(wavBuffer, 0, wavBuffer.Length)) > 0)
                for (int i = 0; i < wavRead; i++) wavSamples.Add(wavBuffer[i]);

            return (wavSamples.ToArray(), wavSampleRate, wavChannels);
        }

        using var reader = new AudioFileReader(path);
        int sampleRate = reader.WaveFormat.SampleRate;
        int channels   = reader.WaveFormat.Channels;

        var list   = new List<float>(sampleRate * channels * 10);
        var buffer = new float[8192];
        int read;
        while ((read = reader.Read(buffer, 0, buffer.Length)) > 0)
            for (int i = 0; i < read; i++) list.Add(buffer[i]);

        return (list.ToArray(), sampleRate, channels);
    }

    /// <summary>ASR target sample rate (same as Config.SampleRate, exposed for external callers).</summary>
    public const int AsrSampleRate = Config.SampleRate;

    /// <summary>
    /// Downmix interleaved multi-channel audio to mono by averaging channels.
    /// If channels == 1, returns the input array directly (no copy).
    /// </summary>
    public static float[] DownmixToMono(float[] audio, int channels)
    {
        if (channels == 1) return audio;
        float[] mono = new float[audio.Length / channels];
        for (int i = 0; i < mono.Length; i++)
        {
            float sum = 0f;
            for (int c = 0; c < channels; c++)
                sum += audio[i * channels + c];
            mono[i] = sum / channels;
        }
        return mono;
    }

    /// <summary>
    /// Downmix to mono and resample to 16 kHz.
    /// Mirrors utils.py audio_to_16000_mono().
    /// </summary>
    public static float[] AudioTo16000Mono(float[] audio, int sampleRate, int channels)
    {
        float[] mono = DownmixToMono(audio, channels);

        if (sampleRate == Config.SampleRate)
            return mono;

        var monoFormat   = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1);
        var waveProvider = new FloatArraySampleProvider(mono, monoFormat);
        var resampler    = new WdlResamplingSampleProvider(waveProvider, Config.SampleRate);

        var outList   = new List<float>((int)((long)mono.Length * Config.SampleRate / sampleRate + 1024));
        var outBuffer = new float[8192];
        int outRead;
        while ((outRead = resampler.Read(outBuffer, 0, outBuffer.Length)) > 0)
            for (int i = 0; i < outRead; i++) outList.Add(outBuffer[i]);

        return outList.ToArray();
    }

    /// <summary>Converts seconds to HH:MM:SS string.</summary>
    public static string SecondsToHhMmSs(double seconds)
        => TimeSpan.FromSeconds(seconds).ToString();
}

// ── Helper: wrap a float[] as an ISampleProvider ─────────────────────────────

public sealed class FloatArraySampleProvider : ISampleProvider
{
    private readonly float[] _data;
    private int _position;

    public FloatArraySampleProvider(float[] data, WaveFormat format)
    {
        _data      = data;
        WaveFormat = format;
    }

    public WaveFormat WaveFormat { get; }

    public int Read(float[] buffer, int offset, int count)
    {
        int available = Math.Min(count, _data.Length - _position);
        if (available <= 0) return 0;
        Array.Copy(_data, _position, buffer, offset, available);
        _position += available;
        return available;
    }
}
