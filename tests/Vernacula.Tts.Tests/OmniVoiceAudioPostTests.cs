using Xunit;
using Vernacula.Tts.Base;

namespace Vernacula.Tts.Tests;

/// <summary>
/// Which finishing chain a render gets is a loudness decision, and the wrong one is audible: a
/// stored voice carries its source clip's own level (the corpus spans rms 0.0017-0.099), so
/// Python's un-boost -- which exists to undo a boost applied at encode time -- makes quiet voices
/// quieter still.
/// </summary>
public class OmniVoiceAudioPostTests
{
    private const int Sr = 24000;

    /// <summary>Half a second of tone at a given peak, loud enough never to read as silence.</summary>
    private static float[] Tone(float peak)
    {
        var a = new float[Sr / 2];
        for (var i = 0; i < a.Length; i++) a[i] = peak * MathF.Sin(2 * MathF.PI * 220 * i / Sr);
        return a;
    }

    private static float Peak(float[] a)
    {
        float m = 0;
        foreach (var v in a) m = MathF.Max(m, MathF.Abs(v));
        return m;
    }

    [Fact]
    public void StoredVoiceChainNormalizesRegardlessOfHowQuietTheReferenceWas()
    {
        // af's reference is rms 0.0148. Under the un-boost that was 15% of level; the render is
        // now normalised, so a quiet source voice is as loud as a loud one.
        var quiet = OmniVoiceAudioPost.FinishStoredVoice(Tone(0.05f), Sr);
        var loud = OmniVoiceAudioPost.FinishStoredVoice(Tone(0.9f), Sr);
        Assert.Equal(0.5f, Peak(quiet), 2);
        Assert.Equal(0.5f, Peak(loud), 2);
    }

    [Fact]
    public void ChunksOfOneDocumentKeepTheirLoudnessRelativeToEachOther()
    {
        // A document is synthesized in chunks and concatenated. Normalising each to 0.5 on its own
        // makes a soft chunk as loud as a shouted one -- audible pumping mid-paragraph. With one
        // leveler, the first chunk anchors the level and a half-as-loud chunk stays half as loud.
        var leveler = new OmniVoiceAudioPost.StoredVoiceLeveler();
        var first = OmniVoiceAudioPost.FinishStoredVoice(Tone(0.4f), Sr, leveler);
        var softer = OmniVoiceAudioPost.FinishStoredVoice(Tone(0.2f), Sr, leveler);
        Assert.Equal(0.5f, Peak(first), 2);
        Assert.Equal(0.25f, Peak(softer), 2);
    }

    [Fact]
    public void ALouderLaterChunkIsHeldShortOfFullScale()
    {
        var leveler = new OmniVoiceAudioPost.StoredVoiceLeveler();
        OmniVoiceAudioPost.FinishStoredVoice(Tone(0.2f), Sr, leveler);
        var louder = OmniVoiceAudioPost.FinishStoredVoice(Tone(0.9f), Sr, leveler);
        Assert.True(Peak(louder) <= 0.95f, $"peak {Peak(louder)} must stay short of clipping");
        Assert.True(Peak(louder) > 0.5f, "and still be louder than the anchor chunk");
    }

    [Fact]
    public void TheDecodedAudioIsNotModified()
    {
        // Both chains leave the caller's buffer alone, so raw decoder output stays raw.
        var raw = Tone(0.3f);
        var before = (float[])raw.Clone();
        OmniVoiceAudioPost.FinishStoredVoice(raw, Sr);
        Assert.Equal(before, raw);
    }

    [Fact]
    public void PythonChainStillUnBoostsAnEncodedReference()
    {
        // Unchanged for a reference WE boosted at encode time: that boost still has to come off.
        var got = OmniVoiceAudioPost.Finish(Tone(0.5f), Sr, refRms: 0.05f);
        Assert.True(Peak(got) < 0.3f, $"expected the un-boost to attenuate, got peak {Peak(got)}");
    }

    [Fact]
    public void PythonChainWithoutAReferencePeakNormalizes()
    {
        var got = OmniVoiceAudioPost.Finish(Tone(0.05f), Sr, refRms: null);
        Assert.Equal(0.5f, Peak(got), 2);
    }
}
