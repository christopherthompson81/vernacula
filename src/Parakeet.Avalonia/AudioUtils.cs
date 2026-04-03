using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using ClosedXML.Excel;
using Parakeet.Base;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace ParakeetCSharp;

internal static class AudioUtils
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

        // FFT bin centre frequencies
        var fftFreqs = new double[freqBins];
        for (int k = 0; k < freqBins; k++)
            fftFreqs[k] = (double)k * Config.SampleRate / Config.NFft;

        // Evenly spaced mel points
        double fminMel = HzToMelSlaney(0.0);
        double fmaxMel = HzToMelSlaney(Config.SampleRate / 2.0);
        var melF = new double[Config.NMels + 2];
        for (int i = 0; i <= Config.NMels + 1; i++)
        {
            double m = fminMel + (fmaxMel - fminMel) * i / (Config.NMels + 1);
            melF[i] = MelToHzSlaney(m);
        }

        // fdiff = np.diff(melF)
        var fdiff = new double[Config.NMels + 1];
        for (int i = 0; i <= Config.NMels; i++)
            fdiff[i] = melF[i + 1] - melF[i];

        // Triangular filters
        for (int i = 0; i < Config.NMels; i++)
        {
            for (int k = 0; k < freqBins; k++)
            {
                double lower = (fftFreqs[k] - melF[i])     / fdiff[i];
                double upper = (melF[i + 2] - fftFreqs[k]) / fdiff[i + 1];
                fb[i, k] = (float)Math.Max(0.0, Math.Min(lower, upper));
            }
        }

        // Slaney normalisation: multiply each row by 2/(mel_f[i+2]-mel_f[i])
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

        // Hann window, zero-padded into n_fft-length array
        // np.hanning(win_length) in Python gives a symmetric window of length win_length
        // (equivalent to scipy.signal.hann(win_length, sym=True) but without the trailing zero)
        // MathNet Window.Hann(n) = periodic, so we use n+1 and drop the last sample to match np.hanning.
        double[] hann = Window.HannPeriodic(winLength); // length winLength, periodic Hann
        int winOffset = (nFft - winLength) / 2;
        var fftWindow = new double[nFft]; // zero-initialized
        for (int i = 0; i < winLength; i++)
            fftWindow[winOffset + i] = hann[i];

        // Centre-pad signal by n_fft/2 on each side
        int pad     = nFft / 2;
        int padded_len = signal.Length + 2 * pad;
        var padded  = new float[padded_len];
        Array.Copy(signal, 0, padded, pad, signal.Length);

        int nFrames = (padded_len - nFft) / hopLength + 1;
        var spec    = new float[freqBins, nFrames];

        var frame = new Complex32[nFft];

        for (int i = 0; i < nFrames; i++)
        {
            int start = i * hopLength;
            // Fill complex frame with windowed samples
            for (int j = 0; j < nFft; j++)
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

    /// <summary>
    /// Full log-mel spectrogram pipeline.  Mirrors utils.py log_mel_spectrogram().
    /// Returns shape (1, T, NMels).
    /// </summary>
    public static float[,,] LogMelSpectrogram(float[] signal)
    {
        float[] y   = Preemphasis(signal);
        float[,] sp = Stft(y);                  // (257, nFrames)

        int freqBins = sp.GetLength(0);
        int nFrames  = sp.GetLength(1);

        // mel_spec = MelFilterbank @ sp  →  (NMels, nFrames)
        var melSpec = new float[Config.NMels, nFrames];
        for (int m = 0; m < Config.NMels; m++)
            for (int t = 0; t < nFrames; t++)
            {
                float sum = 0f;
                for (int k = 0; k < freqBins; k++)
                    sum += MelFilterbank[m, k] * sp[k, t];
                melSpec[m, t] = sum;
            }

        // log(mel + guard), transpose to (1, T, NMels)
        var result = new float[1, nFrames, Config.NMels];
        for (int t = 0; t < nFrames; t++)
            for (int m = 0; m < Config.NMels; m++)
                result[0, t, m] = (float)Math.Log(melSpec[m, t] + Config.LogZeroGuard);

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

    // ── Audio I/O ────────────────────────────────────────────────────────────

    /// <summary>
    /// Read an audio file.  Returns interleaved float samples in [-1, 1],
    /// the sample rate, and the channel count.
    /// <para>
    /// When <paramref name="streamIndex"/> is ≥ 0, or the file extension
    /// belongs to a video container or an FFmpeg-only audio format, decoding
    /// is delegated to <see cref="FFmpegDecoder"/>.  All other files are read
    /// via NAudio (the existing behaviour).
    /// </para>
    /// </summary>
    public static (float[] samples, int sampleRate, int channels)
        ReadAudio(string path, int streamIndex = -1)
    {
        string ext = Path.GetExtension(path);
        bool useFFmpeg = streamIndex >= 0
                      || FFmpegDecoder.VideoExtensions.Contains(ext)
                      || FFmpegDecoder.FfmpegAudioExtensions.Contains(ext);

        if (useFFmpeg)
            return FFmpegDecoder.DecodeStream(path, streamIndex >= 0 ? streamIndex : 0);

        // ── NAudio path (WAV, MP3, FLAC, M4A, OGG, AAC) ──────────────────────
        using var reader = new AudioFileReader(path);
        int sampleRate = reader.WaveFormat.SampleRate;
        int channels   = reader.WaveFormat.Channels;

        var list   = new List<float>(reader.WaveFormat.SampleRate * channels * 10);
        var buffer = new float[8192];
        int read;
        while ((read = reader.Read(buffer, 0, buffer.Length)) > 0)
        {
            for (int i = 0; i < read; i++)
                list.Add(buffer[i]);
        }

        return (list.ToArray(), sampleRate, channels);
    }

    /// <summary>
    /// Downmix to mono and resample to 16 kHz.
    /// Mirrors utils.py audio_to_16000_mono().
    /// </summary>
    public static float[] AudioTo16000Mono(float[] audio, int sampleRate, int channels)
    {
        // ── Downmix to mono ──────────────────────────────────────────────────
        float[] mono;
        if (channels == 1)
        {
            mono = audio;
        }
        else
        {
            mono = new float[audio.Length / channels];
            for (int i = 0; i < mono.Length; i++)
            {
                float sum = 0f;
                for (int c = 0; c < channels; c++)
                    sum += audio[i * channels + c];
                mono[i] = sum / channels;
            }
        }

        // ── Resample to 16 kHz if needed ─────────────────────────────────────
        if (sampleRate == Config.SampleRate)
            return mono;

        // Wrap in a WaveProvider so WdlResamplingSampleProvider can consume it
        var monoFormat   = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, 1);
        var waveProvider = new FloatArraySampleProvider(mono, monoFormat);
        var resampler    = new WdlResamplingSampleProvider(waveProvider, Config.SampleRate);

        var outList   = new List<float>((int)((long)mono.Length * Config.SampleRate / sampleRate + 1024));
        var outBuffer = new float[8192];
        int outRead;
        while ((outRead = resampler.Read(outBuffer, 0, outBuffer.Length)) > 0)
        {
            for (int i = 0; i < outRead; i++)
                outList.Add(outBuffer[i]);
        }

        return outList.ToArray();
    }

    // ── XLSX export ──────────────────────────────────────────────────────────

    /// <summary>
    /// Mirrors utils.py write_xlsx().
    /// rows: list of {column_name → value} dictionaries.
    /// </summary>
    public static void WriteXlsx(IReadOnlyList<IReadOnlyDictionary<string, string>> rows, string outPath)
    {
        if (rows.Count == 0)
            throw new InvalidOperationException("No data found in transcript view.");

        using var wb = new XLWorkbook();
        var ws = wb.Worksheets.Add("Transcript");

        // Collect column order from first row
        var columns = new List<string>(rows[0].Keys);

        // Header row
        for (int col = 0; col < columns.Count; col++)
        {
            var cell = ws.Cell(1, col + 1);
            cell.Value = columns[col];
            cell.Style.Font.Bold = true;
            cell.Style.Fill.BackgroundColor = XLColor.FromHtml("#F0F0F0");
            cell.Style.Alignment.Horizontal = XLAlignmentHorizontalValues.Center;
            cell.Style.Alignment.Vertical   = XLAlignmentVerticalValues.Center;
            cell.Style.Border.OutsideBorder  = XLBorderStyleValues.Thin;
        }

        // Data rows
        int contentColIdx = columns.FindIndex(c => c.Equals("content", StringComparison.OrdinalIgnoreCase));

        for (int rowIdx = 0; rowIdx < rows.Count; rowIdx++)
        {
            for (int col = 0; col < columns.Count; col++)
            {
                var cell = ws.Cell(rowIdx + 2, col + 1);
                cell.Value = rows[rowIdx].TryGetValue(columns[col], out var v) ? v : string.Empty;
                cell.Style.Border.OutsideBorder = XLBorderStyleValues.Thin;

                if (col == contentColIdx)
                    cell.Style.Alignment.WrapText = true;
            }
        }

        // Column widths
        for (int col = 0; col < columns.Count; col++)
        {
            if (col == contentColIdx)
            {
                ws.Column(col + 1).Width = 85;
            }
            else
            {
                ws.Column(col + 1).AdjustToContents();
            }
        }

        wb.SaveAs(outPath);
    }

    /// <summary>Converts seconds to HH:MM:SS string. Mirrors seconds_to_hhmmss_timedelta().</summary>
    public static string SecondsToHhMmSs(double seconds)
        => TimeSpan.FromSeconds(seconds).ToString();
}

// ── Helper: wrap a float[] as an ISampleProvider ─────────────────────────────

internal sealed class FloatArraySampleProvider : ISampleProvider
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
