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
    /// Read an audio file. Returns interleaved float samples in [-1, 1], the sample rate,
    /// and the channel count, all in the decoder's native layout.
    /// <para>
    /// WAV, MP3, AIFF, Ogg Vorbis and Ogg Opus are decoded in-process, on every platform,
    /// with no FFmpeg installed. Everything else — FLAC, M4A, AAC, WMA, non-PCM WAV (mu-law,
    /// A-law, ADPCM) and every video container — is decoded by shelling out to FFmpeg via
    /// <see cref="FfmpegAudioDecoder"/>. <see cref="ManagedAudioDecoders"/> is the one table
    /// that decides which is which.
    /// </para>
    /// <para>
    /// ⚠ THE FFMPEG PATH IS NOT A WINDOWS FALLBACK — IT IS THE ONLY PATH FOR THOSE FORMATS
    /// ON EVERY PLATFORM. The FLAC/M4A/AAC and non-PCM-WAV decoders are MediaFoundation and
    /// ACM P/Invokes living in NAudio.WinMM/NAudio.Wasapi, which NAudio 3 ships only to a
    /// Windows target framework. This project is net10.0, so it has none of them on any
    /// host. Under NAudio 2.3.0 a net10.0 resolve still received those assemblies, so these
    /// formats worked here on Windows and threw on Linux; that asymmetry is what #156
    /// removed, by giving both platforms the FFmpeg route rather than restoring a
    /// Windows-only one.
    /// </para>
    /// <para>
    /// ⚠ AND #156's ANSWER IS WHY THE MANAGED TABLE EXISTS. Routing MP3 to FFmpeg made a
    /// working Windows install stop reading MP3 until FFmpeg was installed — something it
    /// had never needed (#176). Every format with a maintained pure-managed decoder is now
    /// in the table instead, which both restores Windows and adds Linux and macOS, where
    /// several of them never worked in-process at all. FFmpeg remains the fallback for a
    /// file whose bytes turn out not to match its extension.
    /// </para>
    /// <para>
    /// To pick a specific audio stream out of a multi-stream file, call
    /// <see cref="FfmpegAudioDecoder.Decode"/> directly.
    /// </para>
    /// </summary>
    public static (float[] samples, int sampleRate, int channels) ReadAudio(string path)
    {
        if (!ManagedAudioDecoders.TryGet(Path.GetExtension(path), out var decoder))
            return FfmpegAudioDecoder.Decode(path);

        try
        {
            return decoder.Decode(path);
        }
        catch (Exception ex) when (ManagedAudioDecoders.IsFormatRejection(ex))
        {
            // The bytes do not match the extension — a renamed file, a .wav that is really
            // mu-law or ADPCM, an .ogg carrying Speex. FFmpeg sniffs content rather than
            // trusting the name, so it gets the last word.
            try
            {
                return FfmpegAudioDecoder.Decode(path);
            }
            catch (Exception ffmpegEx)
            {
                // ⚠ CARRY THE MANAGED FAILURE FORWARD. This catch is deliberately wide
                // enough to include the exceptions a *corrupt or truncated* file of the
                // right format raises, not just a mislabelled one. Reporting only the
                // FFmpeg error there would hide that the file was rejected as its own
                // format first, which is usually the real diagnosis — and on Windows it
                // sends the user off installing FFmpeg for a file that is simply broken.
                throw new InvalidOperationException(
                    $"Could not read '{Path.GetFileName(path)}'. It was rejected as "
                    + $"{decoder.Name} ({ex.GetType().Name}: {ex.Message}), and the FFmpeg "
                    + $"fallback also failed: {ffmpegEx.Message}", ffmpegEx);
            }
        }
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
    /// Downmix to mono and resample to 16 kHz, then apply always-on audio cleanup
    /// (high-pass at 75 Hz + 50/60 Hz mains-hum notches).  The cleanup targets
    /// specific interferers that trip up ASR — sub-audible rumble, handling noise,
    /// HVAC low-end, mains hum — without touching speech content above ~80 Hz.
    /// </summary>
    public static float[] AudioTo16000Mono(float[] audio, int sampleRate, int channels)
    {
        float[] mono = DownmixToMono(audio, channels);

        float[] at16k;
        if (sampleRate == Config.SampleRate)
        {
            // DownmixToMono returns the input array itself for mono inputs; clone
            // before in-place cleanup so we don't mutate the caller's buffer.
            at16k = ReferenceEquals(mono, audio) ? (float[])mono.Clone() : mono;
        }
        else
        {
            var monoFormat   = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1);
            var waveProvider = new FloatArraySampleProvider(mono, monoFormat);
            var resampler    = new WdlResamplingSampleProvider(waveProvider, Config.SampleRate);

            var outList   = new List<float>((int)((long)mono.Length * Config.SampleRate / sampleRate + 1024));
            var outBuffer = new float[8192];
            int outRead;
            while ((outRead = resampler.Read(outBuffer)) > 0)
                for (int i = 0; i < outRead; i++) outList.Add(outBuffer[i]);
            at16k = outList.ToArray();
        }

        ApplyCleanup(at16k, Config.SampleRate);
        return at16k;
    }

    /// <summary>
    /// Resample a mono float buffer from <paramref name="srcSampleRate"/> to
    /// <paramref name="dstSampleRate"/> (WDL resampler). Returns the input unchanged when the
    /// rates match. The output-capacity estimate is computed in 64-bit to avoid int overflow on
    /// long clips (e.g. a multi-second 16 kHz buffer × a 24 kHz target).
    /// </summary>
    public static float[] ResampleMono(float[] mono, int srcSampleRate, int dstSampleRate)
    {
        if (srcSampleRate == dstSampleRate) return mono;
        var fmt = WaveFormat.CreateIeeeFloatWaveFormat(srcSampleRate, 1);
        var src = new FloatArraySampleProvider(mono, fmt);
        var resampler = new WdlResamplingSampleProvider(src, dstSampleRate);
        var outList = new List<float>(
            (int)((long)mono.Length * dstSampleRate / Math.Max(1, srcSampleRate) + 1024));
        var buf = new float[8192];
        int read;
        while ((read = resampler.Read(buf)) > 0)
            for (int i = 0; i < read; i++) outList.Add(buf[i]);
        return outList.ToArray();
    }

    /// <summary>
    /// In-place always-on preprocessor for ASR input:
    ///   • 2nd-order high-pass at 75 Hz (Butterworth, Q≈0.707) — removes rumble,
    ///     handling noise, and HVAC low-end without touching speech fundamentals
    ///     (typical male voice bottoms out around 80 Hz).
    ///   • Twin narrow notches at 50 Hz and 60 Hz (Q=30) — removes mains hum
    ///     and its spectral fingerprint without audibly affecting nearby speech.
    /// Biquads use RBJ audio-EQ cookbook coefficients, direct form II transposed.
    /// </summary>
    public static void ApplyCleanup(float[] samples, int sampleRate)
    {
        var hp   = Biquad.HighPass(sampleRate, cutoffHz: 75f,  q: 0.7071f);
        var n50  = Biquad.Notch   (sampleRate, centerHz: 50f,  q: 30f);
        var n60  = Biquad.Notch   (sampleRate, centerHz: 60f,  q: 30f);
        for (int i = 0; i < samples.Length; i++)
        {
            float y = hp.Process(samples[i]);
            y = n50.Process(y);
            y = n60.Process(y);
            samples[i] = y;
        }
    }

    /// <summary>
    /// Stateful 2nd-order biquad (direct form II transposed).  Coefficients are
    /// from the RBJ audio-EQ cookbook, normalised by a0 at construction.
    /// </summary>
    private struct Biquad
    {
        private float _b0, _b1, _b2, _a1, _a2;
        private float _z1, _z2;

        public static Biquad HighPass(int fs, float cutoffHz, float q)
        {
            float w0 = 2f * MathF.PI * cutoffHz / fs;
            float cw = MathF.Cos(w0);
            float sw = MathF.Sin(w0);
            float a  = sw / (2f * q);
            float a0 = 1f + a;
            return new Biquad
            {
                _b0 = (1f + cw) / 2f / a0,
                _b1 = -(1f + cw)     / a0,
                _b2 = (1f + cw) / 2f / a0,
                _a1 = -2f * cw       / a0,
                _a2 = (1f - a)       / a0,
            };
        }

        public static Biquad Notch(int fs, float centerHz, float q)
        {
            float w0 = 2f * MathF.PI * centerHz / fs;
            float cw = MathF.Cos(w0);
            float sw = MathF.Sin(w0);
            float a  = sw / (2f * q);
            float a0 = 1f + a;
            return new Biquad
            {
                _b0 = 1f               / a0,
                _b1 = -2f * cw         / a0,
                _b2 = 1f               / a0,
                _a1 = -2f * cw         / a0,
                _a2 = (1f - a)         / a0,
            };
        }

        public float Process(float x)
        {
            float y = _b0 * x + _z1;
            _z1 = _b1 * x - _a1 * y + _z2;
            _z2 = _b2 * x - _a2 * y;
            return y;
        }
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

    // NAudio 3 replaced ISampleProvider.Read(float[], int, int) with Read(Span<float>);
    // the offset the old signature carried is now the caller's slice.
    public int Read(Span<float> buffer)
    {
        int available = Math.Min(buffer.Length, _data.Length - _position);
        if (available <= 0) return 0;
        _data.AsSpan(_position, available).CopyTo(buffer);
        _position += available;
        return available;
    }
}
