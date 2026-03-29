namespace Parakeet.Base;

/// <summary>
/// Powerset decoder for multi-speaker diarization.
/// 
/// The powerset representation encodes all possible speaker combinations as binary masks.
/// For N speakers, there are 2^N - 1 non-empty combinations.
/// 
/// Example with 4 speakers:
/// - powerset[0] = {0} (speaker 0 only)
/// - powerset[1] = {1} (speaker 1 only)  
/// - powerset[2] = {0,1} (speakers 0 and 1 overlapping)
/// - etc.
/// 
/// This decoder converts powerset scores back to per-speaker activity.
/// </summary>
public static class PowersetDecoder
{
    private static Dictionary<int, List<int>[]>? _caches;

    /// <summary>
    /// Get powerset combinations for a given number of speakers.
    /// </summary>
    /// <param name="numSpeakers">Maximum number of speakers (typically 4)</param>
    /// <returns>Array where index i contains the speaker IDs in combination i</returns>
    public static List<int>[] GetPowersetCombinations(int numSpeakers)
    {
        if (_caches == null || !_caches.ContainsKey(numSpeakers))
        {
            var combinations = new List<int>[1 << numSpeakers]; // 2^N
            int count = 0;

            // Generate all non-empty subsets
            for (int mask = 1; mask < (1 << numSpeakers); mask++)
            {
                var speakers = new List<int>();
                for (int s = 0; s < numSpeakers; s++)
                {
                    if ((mask & (1 << s)) != 0)
                        speakers.Add(s);
                }
                combinations[count++] = speakers;
            }

            // Trim to actual count
            var trimmed = new List<int>[count];
            Array.Copy(combinations, trimmed, count);
            
            _caches ??= new Dictionary<int, List<int>[]>();
            _caches[numSpeakers] = trimmed;
        }

        return _caches[numSpeakers];
    }

    /// <summary>
    /// Decode powerset scores to per-speaker activity probabilities.
    /// 
    /// Uses the "powerset to one-hot" transformation from pyannote.
    /// </summary>
    /// <param name="powersetScores">Scores for each powerset combination (frames x combinations)</param>
    /// <param name="numSpeakers">Number of speakers</param>
    /// <returns>Per-speaker activity probabilities (frames x speakers)</returns>
    public static float[][] DecodeToSpeakerActivity(float[] powersetScores, int numSpeakers)
    {
        var combinations = GetPowersetCombinations(numSpeakers);
        int numFrames = powersetScores.Length / combinations.Length;
        var speakerActivity = new float[numFrames][];

        for (int t = 0; t < numFrames; t++)
        {
            var activity = new float[numSpeakers];

            // Sum contributions from all combinations
            for (int c = 0; c < combinations.Length; c++)
            {
                int idx = t * combinations.Length + c;
                if (idx >= powersetScores.Length)
                    break;

                float score = powersetScores[idx];
                if (score > 0)
                {
                    foreach (int speaker in combinations[c])
                        activity[speaker] += score;
                }
            }

            // Normalize to get probabilities
            float sum = activity.Sum();
            if (sum > 0)
            {
                for (int s = 0; s < numSpeakers; s++)
                    activity[s] /= sum;
            }

            speakerActivity[t] = activity;
        }

        return speakerActivity;
    }

    /// <summary>
    /// Binarize powerset scores and extract speaker assignments.
    /// </summary>
    /// <param name="powersetScores">Powerset scores (frames x combinations)</param>
    /// <param name="numSpeakers">Number of speakers</param>
    /// <param name="threshold">Binarization threshold</param>
    /// <returns>Array of (frame, List of active speakers)</returns>
    public static List<int>[][] BinarizePowerset(
        float[] powersetScores,
        int numSpeakers,
        float threshold = 0.5f)
    {
        var combinations = GetPowersetCombinations(numSpeakers);
        int numFrames = powersetScores.Length / combinations.Length;
        var activeSpeakers = new List<int>[numFrames];

        for (int t = 0; t < numFrames; t++)
        {
            activeSpeakers[t] = new List<int>();

            // Find best matching combination
            int bestCombo = -1;
            float bestScore = -float.MaxValue;

            for (int c = 0; c < combinations.Length; c++)
            {
                int idx = t * combinations.Length + c;
                if (idx >= powersetScores.Length)
                    break;

                if (powersetScores[idx] > bestScore)
                {
                    bestScore = powersetScores[idx];
                    bestCombo = c;
                }
            }

            // If best combination passes threshold, assign those speakers
            if (bestScore >= threshold && bestCombo >= 0)
            {
                activeSpeakers[t] = new List<int>(combinations[bestCombo]);
            }
        }

        return activeSpeakers;
    }

    /// <summary>
    /// Count maximum simultaneous speakers from powerset combinations.
    /// </summary>
    public static int GetMaxSimultaneousSpeakers(int numSpeakers)
    {
        var combinations = GetPowersetCombinations(numSpeakers);
        return combinations.Max(c => c.Count);
    }

    /// <summary>
    /// Get the number of powerset combinations for N speakers.
    /// </summary>
    public static int GetNumCombinations(int numSpeakers)
    {
        return (1 << numSpeakers) - 1; // 2^N - 1
    }
}
