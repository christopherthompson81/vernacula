// Audio and mel-spectrogram utilities for the benchmark.
// Mirrors AudioUtils.cs from the main project (no WPF or ClosedXML deps).

using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

internal static class BenchmarkAudioUtils
{
    // ── Constants (mirrors Config.cs) ─────────────────────────────────────────

    public const int   SampleRate   = 16_000;
    public const int   NFft         = 512;
    public const int   WinLength    = 400;
    public const int   HopLength    = 160;
    public const int   NMels        = 128;
    public const float Preemph      = 0.97f;
    public const float LogZeroGuard = 5.960464478e-8f;

    // Slaney mel-scale parameters
    private const  double FMin     = 0.0;
    private const  double FSp      = 200.0 / 3.0;
    private const  double MinLogHz = 1000.0;
    private static readonly double MinLogMel = (1000.0 - FMin) / FSp;
    private static readonly double LogStep   = Math.Log(6.4) / 27.0;

    // ── Mel filterbank (computed once on first use) ───────────────────────────

    public static readonly float[,] MelFilterbank = CreateMelFilterbank();

    private static double HzToMelSlaney(double hz) =>
        hz >= MinLogHz
            ? MinLogMel + Math.Log(hz / MinLogHz) / LogStep
            : (hz - FMin) / FSp;

    private static double MelToHzSlaney(double mel) =>
        mel >= MinLogMel
            ? MinLogHz * Math.Exp(LogStep * (mel - MinLogMel))
            : FMin + FSp * mel;

    private static float[,] CreateMelFilterbank()
    {
        int freqBins = NFft / 2 + 1;
        var fb       = new float[NMels, freqBins];

        var fftFreqs = new double[freqBins];
        for (int k = 0; k < freqBins; k++)
            fftFreqs[k] = (double)k * SampleRate / NFft;

        double fminMel = HzToMelSlaney(0.0);
        double fmaxMel = HzToMelSlaney(SampleRate / 2.0);
        var melF = new double[NMels + 2];
        for (int i = 0; i <= NMels + 1; i++)
        {
            double m = fminMel + (fmaxMel - fminMel) * i / (NMels + 1);
            melF[i] = MelToHzSlaney(m);
        }

        var fdiff = new double[NMels + 1];
        for (int i = 0; i <= NMels; i++) fdiff[i] = melF[i + 1] - melF[i];

        for (int i = 0; i < NMels; i++)
            for (int k = 0; k < freqBins; k++)
            {
                double lower = (fftFreqs[k] - melF[i])     / fdiff[i];
                double upper = (melF[i + 2] - fftFreqs[k]) / fdiff[i + 1];
                fb[i, k] = (float)Math.Max(0.0, Math.Min(lower, upper));
            }

        for (int i = 0; i < NMels; i++)
        {
            float enorm = (float)(2.0 / (melF[i + 2] - melF[i]));
            for (int k = 0; k < freqBins; k++) fb[i, k] *= enorm;
        }

        return fb;
    }

    // ── Feature extraction ────────────────────────────────────────────────────

    public static float[] Preemphasis(float[] signal)
    {
        var out_ = new float[signal.Length];
        out_[0] = signal[0];
        for (int i = 1; i < signal.Length; i++)
            out_[i] = signal[i] - Preemph * signal[i - 1];
        return out_;
    }

    public static float[,] Stft(float[] signal)
    {
        int freqBins = NFft / 2 + 1;

        double[] hann      = Window.HannPeriodic(WinLength);
        int      winOffset = (NFft - WinLength) / 2;
        var      fftWindow = new double[NFft];
        for (int i = 0; i < WinLength; i++) fftWindow[winOffset + i] = hann[i];

        int pad       = NFft / 2;
        int paddedLen = signal.Length + 2 * pad;
        var padded    = new float[paddedLen];
        Array.Copy(signal, 0, padded, pad, signal.Length);

        int nFrames = (paddedLen - NFft) / HopLength + 1;
        var spec    = new float[freqBins, nFrames];
        var frame   = new Complex32[NFft];

        for (int i = 0; i < nFrames; i++)
        {
            int start = i * HopLength;
            for (int j = 0; j < NFft; j++)
                frame[j] = new Complex32((float)(padded[start + j] * fftWindow[j]), 0f);

            Fourier.Forward(frame, FourierOptions.NoScaling);

            for (int k = 0; k < freqBins; k++)
            {
                float re = frame[k].Real;
                float im = frame[k].Imaginary;
                spec[k, i] = re * re + im * im;
            }
        }

        return spec;
    }

    /// <summary>Returns shape (1, T, NMels).</summary>
    public static float[,,] LogMelSpectrogram(float[] signal)
    {
        float[] y   = Preemphasis(signal);
        float[,] sp = Stft(y);

        int freqBins = sp.GetLength(0);
        int nFrames  = sp.GetLength(1);

        var melSpec = new float[NMels, nFrames];
        for (int m = 0; m < NMels; m++)
            for (int t = 0; t < nFrames; t++)
            {
                float sum = 0f;
                for (int k = 0; k < freqBins; k++)
                    sum += MelFilterbank[m, k] * sp[k, t];
                melSpec[m, t] = sum;
            }

        var result = new float[1, nFrames, NMels];
        for (int t = 0; t < nFrames; t++)
            for (int m = 0; m < NMels; m++)
                result[0, t, m] = (float)Math.Log(melSpec[m, t] + LogZeroGuard);

        return result;
    }

    // ── Audio I/O ─────────────────────────────────────────────────────────────

    public static float[] LoadAudio16kMono(string path, out double durationSec)
    {
        using var reader = new AudioFileReader(path);
        int sr = reader.WaveFormat.SampleRate;
        int ch = reader.WaveFormat.Channels;

        var list = new List<float>(sr * ch * 120);
        var buf  = new float[8192];
        int read;
        while ((read = reader.Read(buf, 0, buf.Length)) > 0)
            for (int i = 0; i < read; i++) list.Add(buf[i]);

        float[] mono;
        if (ch == 1)
        {
            mono = list.ToArray();
        }
        else
        {
            mono = new float[list.Count / ch];
            for (int i = 0; i < mono.Length; i++)
            {
                float sum = 0f;
                for (int c = 0; c < ch; c++) sum += list[i * ch + c];
                mono[i] = sum / ch;
            }
        }

        float[] result;
        if (sr == SampleRate)
        {
            result = mono;
        }
        else
        {
            var fmt      = WaveFormat.CreateIeeeFloatWaveFormat(sr, 1);
            var provider = new BenchmarkFloatArrayProvider(mono, fmt);
            var resampler = new WdlResamplingSampleProvider(provider, SampleRate);
            var outList  = new List<float>(mono.Length * SampleRate / sr + 1024);
            var outBuf   = new float[8192];
            int n;
            while ((n = resampler.Read(outBuf, 0, outBuf.Length)) > 0)
                for (int i = 0; i < n; i++) outList.Add(outBuf[i]);
            result = outList.ToArray();
        }

        durationSec = result.Length / (double)SampleRate;
        return result;
    }
}

// ── NAudio helper ─────────────────────────────────────────────────────────────

internal sealed class BenchmarkFloatArrayProvider(float[] data, WaveFormat format) : ISampleProvider
{
    private int _pos;
    public WaveFormat WaveFormat { get; } = format;
    public int Read(float[] buffer, int offset, int count)
    {
        int n = Math.Min(count, data.Length - _pos);
        if (n <= 0) return 0;
        Array.Copy(data, _pos, buffer, offset, n);
        _pos += n;
        return n;
    }
}
