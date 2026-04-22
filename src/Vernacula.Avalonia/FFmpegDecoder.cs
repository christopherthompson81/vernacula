using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using FFmpeg.AutoGen;
using Vernacula.App.Models;

namespace Vernacula.App;

/// <summary>
/// Wraps FFmpeg.AutoGen to probe audio streams and decode audio from any
/// container/codec that FFmpeg supports.  Handles video containers (MP4, MKV,
/// MOV, …) and audio codecs that NAudio cannot decode (OPUS, etc.).
/// </summary>
internal static unsafe class FFmpegDecoder
{
    // ── Extension classification ──────────────────────────────────────────────

    /// <summary>File extensions that are video containers (always decoded via FFmpeg).</summary>
    public static readonly HashSet<string> VideoExtensions =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ".mp4", ".mov", ".webm", ".mkv", ".avi", ".wmv", ".flv",
            ".ts",  ".mts", ".m2ts", ".3gp",
        };

    /// <summary>Audio-only extensions that NAudio cannot handle reliably.</summary>
    public static readonly HashSet<string> FfmpegAudioExtensions =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ".opus", ".wma", ".aiff", ".ogg",
        };

    // ── Initialisation ────────────────────────────────────────────────────────

    private static bool _initialized;

    /// <summary>
    /// Points FFmpeg.AutoGen at the native DLL directory.
    /// Must be called once before any other method.
    /// </summary>
    public static void Initialize(string rootPath)
    {
        if (_initialized) return;

        // Prefer runtimes/<rid>/native/ only when it actually contains FFmpeg
        // shared libraries. Other NuGet packages (onnxruntime, Skia, …) populate
        // that directory with their own natives, and pointing ffmpeg.RootPath
        // there silently breaks symbol resolution: the library is not found,
        // stubs stay unbound, and later calls throw NotSupportedException.
        // Falling back to empty RootPath lets the OS linker resolve via ldconfig
        // (Linux), DYLD cache (macOS), or PATH (Windows).
        // FFmpeg.AutoGen initialises RootPath to the AppContext base directory,
        // which is fine when we ship natives in runtimes/<rid>/native, but hides
        // the system-installed FFmpeg from dlopen when we don't. Force it to
        // empty so Linux/macOS/Windows fall back to ldconfig / DYLD / PATH.
        string? runtimesDir = FindNativeRuntimeDirectory(rootPath);
        ffmpeg.RootPath = runtimesDir ?? string.Empty;

        _initialized = true;
    }

    private static string? FindNativeRuntimeDirectory(string rootPath)
    {
        foreach (string rid in EnumerateRuntimeIdentifiers())
        {
            string candidate = Path.Combine(rootPath, "runtimes", rid, "native");
            if (!Directory.Exists(candidate))
                continue;

            if (Directory.EnumerateFiles(candidate, "*avformat*").Any())
                return candidate;
        }

        return null;
    }

    private static IEnumerable<string> EnumerateRuntimeIdentifiers()
    {
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        void Add(string rid)
        {
            if (!string.IsNullOrWhiteSpace(rid))
                seen.Add(rid);
        }

        Add(RuntimeInformation.RuntimeIdentifier);

        string arch = RuntimeInformation.ProcessArchitecture switch
        {
            Architecture.X64 => "x64",
            Architecture.X86 => "x86",
            Architecture.Arm64 => "arm64",
            Architecture.Arm => "arm",
            Architecture.S390x => "s390x",
            Architecture.LoongArch64 => "loongarch64",
            _ => RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant(),
        };

        if (OperatingSystem.IsWindows())
            Add($"win-{arch}");
        else if (OperatingSystem.IsMacOS())
            Add($"osx-{arch}");
        else if (OperatingSystem.IsLinux())
        {
            Add($"linux-{arch}");
            Add($"linux-musl-{arch}");
        }

        return seen;
    }

    // ── Stream probing ────────────────────────────────────────────────────────

    /// <summary>
    /// Returns metadata for every audio stream in the file.
    /// Returns an empty list if the file cannot be opened or contains no audio.
    /// </summary>
    public static List<AudioStreamInfo> ProbeAudioStreams(string filePath)
    {
        var result = new List<AudioStreamInfo>();
        AVFormatContext* fmtCtx = null;

        if (ffmpeg.avformat_open_input(&fmtCtx, filePath, null, null) < 0)
            return result;

        try
        {
            if (ffmpeg.avformat_find_stream_info(fmtCtx, null) < 0)
                return result;

            double containerDuration = fmtCtx->duration != ffmpeg.AV_NOPTS_VALUE
                ? fmtCtx->duration / (double)ffmpeg.AV_TIME_BASE
                : 0;

            for (int i = 0; i < (int)fmtCtx->nb_streams; i++)
            {
                AVStream* stream = fmtCtx->streams[i];
                if (stream->codecpar->codec_type != AVMediaType.AVMEDIA_TYPE_AUDIO)
                    continue;

                AVCodecDescriptor* desc =
                    ffmpeg.avcodec_descriptor_get(stream->codecpar->codec_id);
                string codecName = desc != null
                    ? Marshal.PtrToStringUTF8((IntPtr)desc->name) ?? "unknown"
                    : "unknown";

                string? language = ReadDictEntry(stream->metadata, "language");
                string? title    = ReadDictEntry(stream->metadata, "title");

                int channels   = stream->codecpar->ch_layout.nb_channels;
                int sampleRate = stream->codecpar->sample_rate;

                double duration = stream->duration != ffmpeg.AV_NOPTS_VALUE
                    ? stream->duration * ffmpeg.av_q2d(stream->time_base)
                    : containerDuration;

                result.Add(new AudioStreamInfo(
                    i, codecName, language, title, channels, sampleRate, duration));
            }
        }
        finally
        {
            ffmpeg.avformat_close_input(&fmtCtx);
        }

        return result;
    }

    // ── Audio decoding ────────────────────────────────────────────────────────

    /// <summary>
    /// Decodes the audio stream at <paramref name="streamIndex"/> to
    /// float32 interleaved samples at the stream's native sample rate.
    /// Pass <c>-1</c> (or any invalid index) to decode the first audio stream.
    /// <para>
    /// <see cref="AudioUtils.AudioTo16000Mono"/> is responsible for downmixing
    /// and resampling the returned data to 16 kHz mono.
    /// </para>
    /// </summary>
    public static (float[] samples, int sampleRate, int channels)
        DecodeStream(string filePath, int streamIndex)
    {
        try
        {
            return DecodeStreamAutoGen(filePath, streamIndex);
        }
        catch (NotSupportedException)
        {
            return DecodeStreamViaCli(filePath, streamIndex);
        }
    }

    private static (float[] samples, int sampleRate, int channels)
        DecodeStreamAutoGen(string filePath, int streamIndex)
    {
        AVFormatContext* fmtCtx = null;

        if (ffmpeg.avformat_open_input(&fmtCtx, filePath, null, null) < 0)
            throw new InvalidOperationException($"Failed to open media file: {filePath}");

        try
        {
            if (ffmpeg.avformat_find_stream_info(fmtCtx, null) < 0)
                throw new InvalidOperationException("Failed to read stream info.");

            // Resolve target stream index
            int target = ResolveAudioStream(fmtCtx, streamIndex);

            AVStream*     stream   = fmtCtx->streams[target];
            AVCodecContext* codecCtx = OpenCodec(stream);

            try
            {
                int outChannels   = codecCtx->ch_layout.nb_channels;
                int outSampleRate = codecCtx->sample_rate;

                SwrContext* swrCtx = BuildSwrContext(codecCtx);

                try
                {
                    var samples = DecodeSamples(
                        fmtCtx, codecCtx, swrCtx, target, outChannels);

                    return (samples, outSampleRate, outChannels);
                }
                finally
                {
                    ffmpeg.swr_free(&swrCtx);
                }
            }
            finally
            {
                ffmpeg.avcodec_free_context(&codecCtx);
            }
        }
        finally
        {
            ffmpeg.avformat_close_input(&fmtCtx);
        }
    }

    private static (float[] samples, int sampleRate, int channels)
        DecodeStreamViaCli(string filePath, int streamIndex)
    {
        var stream = ProbeAudioStreamsViaCli(filePath)
            .FirstOrDefault(s => s.StreamIndex == streamIndex);

        if (stream is null)
        {
            stream = ProbeAudioStreamsViaCli(filePath).FirstOrDefault()
                ?? throw new InvalidOperationException("No audio streams found in file.");
        }

        string mapSpecifier = $"0:{stream.StreamIndex}";
        var psi = new ProcessStartInfo("ffmpeg")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        psi.ArgumentList.Add("-v");
        psi.ArgumentList.Add("error");
        psi.ArgumentList.Add("-i");
        psi.ArgumentList.Add(filePath);
        psi.ArgumentList.Add("-map");
        psi.ArgumentList.Add(mapSpecifier);
        psi.ArgumentList.Add("-f");
        psi.ArgumentList.Add("f32le");
        psi.ArgumentList.Add("-acodec");
        psi.ArgumentList.Add("pcm_f32le");
        psi.ArgumentList.Add("-");

        using var process = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start ffmpeg process.");

        using var ms = new MemoryStream();
        process.StandardOutput.BaseStream.CopyTo(ms);
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        if (process.ExitCode != 0)
            throw new InvalidOperationException(
                $"ffmpeg failed to decode audio stream {stream.StreamIndex}: {stderr.Trim()}");

        byte[] raw = ms.ToArray();
        if (raw.Length % sizeof(float) != 0)
            throw new InvalidOperationException("Decoded PCM payload had an unexpected size.");

        float[] samples = new float[raw.Length / sizeof(float)];
        Buffer.BlockCopy(raw, 0, samples, 0, raw.Length);
        return (samples, stream.SampleRate, stream.Channels);
    }

    private static List<AudioStreamInfo> ProbeAudioStreamsViaCli(string filePath)
    {
        var psi = new ProcessStartInfo("ffprobe")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        psi.ArgumentList.Add("-v");
        psi.ArgumentList.Add("error");
        psi.ArgumentList.Add("-show_streams");
        psi.ArgumentList.Add("-select_streams");
        psi.ArgumentList.Add("a");
        psi.ArgumentList.Add("-of");
        psi.ArgumentList.Add("json");
        psi.ArgumentList.Add(filePath);

        using var process = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start ffprobe process.");

        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        if (process.ExitCode != 0)
            throw new InvalidOperationException(
                $"ffprobe failed to inspect media file: {stderr.Trim()}");

        using JsonDocument doc = JsonDocument.Parse(stdout);
        if (!doc.RootElement.TryGetProperty("streams", out JsonElement streamsEl)
            || streamsEl.ValueKind != JsonValueKind.Array)
            return new List<AudioStreamInfo>();

        var streams = new List<AudioStreamInfo>();
        foreach (JsonElement streamEl in streamsEl.EnumerateArray())
        {
            int index = streamEl.TryGetProperty("index", out JsonElement indexEl)
                && indexEl.TryGetInt32(out int parsedIndex)
                ? parsedIndex
                : -1;
            string codec = streamEl.TryGetProperty("codec_name", out JsonElement codecEl)
                ? codecEl.GetString() ?? "unknown"
                : "unknown";
            int channels = streamEl.TryGetProperty("channels", out JsonElement channelsEl)
                && channelsEl.TryGetInt32(out int parsedChannels)
                ? parsedChannels
                : 1;
            int sampleRate = streamEl.TryGetProperty("sample_rate", out JsonElement sampleRateEl)
                && int.TryParse(sampleRateEl.GetString(), out int parsedRate)
                ? parsedRate
                : 0;
            double duration = streamEl.TryGetProperty("duration", out JsonElement durationEl)
                && double.TryParse(durationEl.GetString(), out double parsedDuration)
                ? parsedDuration
                : 0;

            string? language = null;
            string? title = null;
            if (streamEl.TryGetProperty("tags", out JsonElement tagsEl)
                && tagsEl.ValueKind == JsonValueKind.Object)
            {
                if (tagsEl.TryGetProperty("language", out JsonElement langEl))
                    language = langEl.GetString();
                if (tagsEl.TryGetProperty("title", out JsonElement titleEl))
                    title = titleEl.GetString();
            }

            streams.Add(new AudioStreamInfo(
                index, codec, language, title, channels, sampleRate, duration));
        }

        return streams;
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private static int ResolveAudioStream(AVFormatContext* fmtCtx, int requested)
    {
        int nb = (int)fmtCtx->nb_streams;

        // If the caller provided a valid audio stream index, use it
        if (requested >= 0 && requested < nb &&
            fmtCtx->streams[requested]->codecpar->codec_type == AVMediaType.AVMEDIA_TYPE_AUDIO)
            return requested;

        // Otherwise find the first audio stream
        for (int i = 0; i < nb; i++)
            if (fmtCtx->streams[i]->codecpar->codec_type == AVMediaType.AVMEDIA_TYPE_AUDIO)
                return i;

        throw new InvalidOperationException("No audio streams found in file.");
    }

    private static AVCodecContext* OpenCodec(AVStream* stream)
    {
        AVCodec* codec = ffmpeg.avcodec_find_decoder(stream->codecpar->codec_id);
        if (codec == null)
            throw new InvalidOperationException(
                $"No decoder found for codec id {stream->codecpar->codec_id}.");

        AVCodecContext* ctx = ffmpeg.avcodec_alloc_context3(codec);
        if (ctx == null)
            throw new InvalidOperationException("Failed to allocate codec context.");

        if (ffmpeg.avcodec_parameters_to_context(ctx, stream->codecpar) < 0)
        {
            ffmpeg.avcodec_free_context(&ctx);
            throw new InvalidOperationException("Failed to copy codec parameters.");
        }

        if (ffmpeg.avcodec_open2(ctx, codec, null) < 0)
        {
            ffmpeg.avcodec_free_context(&ctx);
            throw new InvalidOperationException("Failed to open codec.");
        }

        return ctx;
    }

    private static SwrContext* BuildSwrContext(AVCodecContext* codecCtx)
    {
        SwrContext* swr = null;

        // Copy channel layout so we can take an address of a local
        AVChannelLayout outLayout = codecCtx->ch_layout;
        AVChannelLayout inLayout  = codecCtx->ch_layout;

        int ret = ffmpeg.swr_alloc_set_opts2(
            &swr,
            &outLayout, AVSampleFormat.AV_SAMPLE_FMT_FLT, codecCtx->sample_rate,
            &inLayout,  codecCtx->sample_fmt,              codecCtx->sample_rate,
            0, null);

        if (ret < 0 || swr == null)
            throw new InvalidOperationException("Failed to create SwrContext.");

        if (ffmpeg.swr_init(swr) < 0)
        {
            ffmpeg.swr_free(&swr);
            throw new InvalidOperationException("Failed to initialise SwrContext.");
        }

        return swr;
    }

    private static float[] DecodeSamples(
        AVFormatContext* fmtCtx,
        AVCodecContext*  codecCtx,
        SwrContext*      swrCtx,
        int              targetStream,
        int              channels)
    {
        var    samples = new List<float>(codecCtx->sample_rate * channels * 10);
        AVPacket* pkt  = ffmpeg.av_packet_alloc();
        AVFrame*  frame = ffmpeg.av_frame_alloc();

        try
        {
            // Read and decode all packets for the target stream
            while (ffmpeg.av_read_frame(fmtCtx, pkt) >= 0)
            {
                if (pkt->stream_index == targetStream)
                    SendAndReceive(codecCtx, swrCtx, pkt, frame, channels, samples);
                ffmpeg.av_packet_unref(pkt);
            }

            // Flush the decoder
            ffmpeg.avcodec_send_packet(codecCtx, null);
            int ret;
            while ((ret = ffmpeg.avcodec_receive_frame(codecCtx, frame)) >= 0)
            {
                ConvertAndAccumulate(swrCtx, frame, channels, samples);
                ffmpeg.av_frame_unref(frame);
            }

            // Flush any samples buffered in the resampler
            FlushSwr(swrCtx, channels, samples);
        }
        finally
        {
            ffmpeg.av_packet_free(&pkt);
            ffmpeg.av_frame_free(&frame);
        }

        return samples.ToArray();
    }

    private static void SendAndReceive(
        AVCodecContext* codecCtx,
        SwrContext*     swrCtx,
        AVPacket*       pkt,
        AVFrame*        frame,
        int             channels,
        List<float>     samples)
    {
        if (ffmpeg.avcodec_send_packet(codecCtx, pkt) < 0) return;

        int ret;
        while ((ret = ffmpeg.avcodec_receive_frame(codecCtx, frame)) >= 0)
        {
            ConvertAndAccumulate(swrCtx, frame, channels, samples);
            ffmpeg.av_frame_unref(frame);
        }
    }

    private static void ConvertAndAccumulate(
        SwrContext* swrCtx,
        AVFrame*    frame,
        int         channels,
        List<float> samples)
    {
        int maxOut = (int)ffmpeg.av_rescale_rnd(
            ffmpeg.swr_get_delay(swrCtx, frame->sample_rate) + frame->nb_samples,
            frame->sample_rate, frame->sample_rate, AVRounding.AV_ROUND_UP);

        if (maxOut <= 0) return;

        byte* outBuf = null;
        int   linesize;
        if (ffmpeg.av_samples_alloc(&outBuf, &linesize, channels, maxOut,
                AVSampleFormat.AV_SAMPLE_FMT_FLT, 0) < 0)
            return;

        try
        {
            int converted = ffmpeg.swr_convert(
                swrCtx, &outBuf, maxOut,
                frame->extended_data, frame->nb_samples);

            if (converted > 0)
            {
                float* data  = (float*)outBuf;
                int    count = converted * channels;
                for (int i = 0; i < count; i++)
                    samples.Add(data[i]);
            }
        }
        finally
        {
            ffmpeg.av_freep(&outBuf);
        }
    }

    private static void FlushSwr(SwrContext* swrCtx, int channels, List<float> samples)
    {
        const int BufSamples = 4096;

        byte* outBuf = null;
        int   linesize;
        if (ffmpeg.av_samples_alloc(&outBuf, &linesize, channels, BufSamples,
                AVSampleFormat.AV_SAMPLE_FMT_FLT, 0) < 0)
            return;

        try
        {
            int flushed;
            while ((flushed = ffmpeg.swr_convert(
                        swrCtx, &outBuf, BufSamples, null, 0)) > 0)
            {
                float* data  = (float*)outBuf;
                int    count = flushed * channels;
                for (int i = 0; i < count; i++)
                    samples.Add(data[i]);
            }
        }
        finally
        {
            ffmpeg.av_freep(&outBuf);
        }
    }

    private static string? ReadDictEntry(AVDictionary* dict, string key)
    {
        AVDictionaryEntry* entry = ffmpeg.av_dict_get(dict, key, null, 0);
        return entry != null
            ? Marshal.PtrToStringUTF8((IntPtr)entry->value)
            : null;
    }
}
