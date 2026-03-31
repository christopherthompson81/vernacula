using System.Runtime.InteropServices;
using FFmpeg.AutoGen;
using ParakeetCSharp.Models;

namespace ParakeetCSharp;

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

        // Modern .NET places native runtime assets in runtimes/{rid}/native/ —
        // check that subdirectory first, fall back to the root output folder.
        string runtimesDir = Path.Combine(rootPath, "runtimes", "win-x64", "native");
        ffmpeg.RootPath = Directory.Exists(runtimesDir) ? runtimesDir : rootPath;

        // avformat_network_init() is only needed for network protocols (HTTP/RTMP/…).
        // We use local-file decoding only, so skip the call to avoid loading DLLs
        // eagerly at startup — they will be loaded on demand on first use instead.

        _initialized = true;
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
