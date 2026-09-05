using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using Vernacula.Base;

namespace Vernacula.Tts.Base.AudioIo;

/// <summary>
/// Reads a voice-prompt WAV from disk and returns a 24 kHz mono float32
/// buffer padded or cropped to the canonical
/// <see cref="ChatterboxConstants.DummyAudioSamples"/> trace length the
/// speech_encoder was exported with.
///
/// Resampling uses NAudio's <see cref="WdlResamplingSampleProvider"/>. The
/// Python reference (<c>listen_test.py</c>) uses librosa's kaiser_best
/// resampler — different samples emerge from the same 16 kHz input, which
/// drifts the speaker embedding and the resulting LM token sequence (see
/// docs/chatterbox_investigation.md / issue #53). For reproducibility
/// across implementations, pre-resample the prompt to 24 kHz with the same
/// tool both pipelines use.
/// </summary>
public static class VoicePromptLoader
{
    public static float[] Load(string path)
    {
        var (raw, sr, channels) = AudioUtils.ReadAudio(path);
        float[] mono = AudioUtils.DownmixToMono(raw, channels);

        float[] at24k;
        if (sr == ChatterboxConstants.S3GenSr)
        {
            at24k = ReferenceEquals(mono, raw) ? (float[])mono.Clone() : mono;
        }
        else
        {
            var srcFmt = WaveFormat.CreateIeeeFloatWaveFormat(sr, 1);
            var provider = new FloatArraySampleProvider(mono, srcFmt);
            var resampler = new WdlResamplingSampleProvider(provider, ChatterboxConstants.S3GenSr);
            var outList = new List<float>((int)((long)mono.Length * ChatterboxConstants.S3GenSr / sr + 1024));
            var buf = new float[8192];
            int n;
            while ((n = resampler.Read(buf, 0, buf.Length)) > 0)
                for (int i = 0; i < n; i++) outList.Add(buf[i]);
            at24k = outList.ToArray();
        }

        if (at24k.Length >= ChatterboxConstants.DummyAudioSamples)
            return at24k.AsSpan(0, ChatterboxConstants.DummyAudioSamples).ToArray();

        var padded = new float[ChatterboxConstants.DummyAudioSamples];
        Array.Copy(at24k, padded, at24k.Length);
        return padded;
    }

    /// <summary>
    /// NAudio float-array source. Local because the matching helper inside
    /// Vernacula.Base is internal — duplicating ~15 LOC beats widening that
    /// API surface for a single consumer.
    /// </summary>
    private sealed class FloatArraySampleProvider : ISampleProvider
    {
        private readonly float[] _data;
        private int _pos;
        public FloatArraySampleProvider(float[] data, WaveFormat fmt) { _data = data; WaveFormat = fmt; }
        public WaveFormat WaveFormat { get; }
        public int Read(float[] buffer, int offset, int count)
        {
            int remain = _data.Length - _pos;
            int take = Math.Min(remain, count);
            if (take <= 0) return 0;
            Array.Copy(_data, _pos, buffer, offset, take);
            _pos += take;
            return take;
        }
    }
}
