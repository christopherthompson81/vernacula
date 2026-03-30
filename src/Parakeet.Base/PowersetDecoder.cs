namespace Parakeet.Base;

/// <summary>
/// Powerset decoder for multi-speaker diarization.
///
/// DiariZen uses a restricted powerset encoding:
/// - max 4 unique speakers per chunk
/// - max 2 simultaneous speakers per frame
/// - This gives C(4,0) + C(4,1) + C(4,2) = 1 + 4 + 6 = 11 classes
///
/// Powerset combinations:
/// - [0,0,0,0] = silence (class 0)
/// - [1,0,0,0] = speaker 0 only (class 1)
/// - [0,1,0,0] = speaker 1 only (class 2)
/// - [0,0,1,0] = speaker 2 only (class 3)
/// - [0,0,0,1] = speaker 3 only (class 4)
/// - [1,1,0,0] = speakers 0+1 (class 5)
/// - [1,0,1,0] = speakers 0+2 (class 6)
/// - [1,0,0,1] = speakers 0+3 (class 7)
/// - [0,1,1,0] = speakers 1+2 (class 8)
/// - [0,1,0,1] = speakers 1+3 (class 9)
/// - [0,0,1,1] = speakers 2+3 (class 10)
/// </summary>
public static class PowersetDecoder
{
    private static Dictionary<string, List<int>[]>? _caches;

    /// <summary>
    /// Get powerset combinations for a given number of speakers.
    /// </summary>
    /// <param name="numSpeakers">Maximum number of unique speakers (typically 4)</param>
    /// <param name="maxSimultaneous">Maximum simultaneous speakers per frame (typically 2)</param>
    /// <returns>Array where index i contains the speaker IDs in combination i</returns>
    public static List<int>[] GetPowersetCombinations(int numSpeakers, int maxSimultaneous = 2)
    {
        var cacheKey = $"{numSpeakers}_{maxSimultaneous}";
        if (_caches == null || !_caches.ContainsKey(cacheKey))
        {
            var allCombinations = new List<List<int>>();

            // Must match pyannote's ordering: iterate k=0,1,...,maxSimultaneous;
            // within each k use itertools.combinations order (lexicographic k-subsets).
            // For N=4, maxSim=2: [], [0],[1],[2],[3], [0,1],[0,2],[0,3],[1,2],[1,3],[2,3]
            var speakers = Enumerable.Range(0, numSpeakers).ToList();
            for (int k = 0; k <= maxSimultaneous; k++)
            {
                foreach (var combo in KCombinations(speakers, k))
                    allCombinations.Add(combo);
            }

            var combinations = allCombinations.ToArray();
            _caches ??= new Dictionary<string, List<int>[]>();
            _caches[cacheKey] = combinations;
        }

        return _caches[cacheKey];
    }

    /// <summary>Generates all k-element subsets of <paramref name="items"/> in lexicographic order.</summary>
    private static IEnumerable<List<int>> KCombinations(List<int> items, int k)
    {
        if (k == 0) { yield return []; yield break; }
        for (int i = 0; i <= items.Count - k; i++)
        {
            foreach (var rest in KCombinations(items.GetRange(i + 1, items.Count - i - 1), k - 1))
            {
                var result = new List<int>(k) { items[i] };
                result.AddRange(rest);
                yield return result;
            }
        }
    }

    /// <summary>
    /// Decode powerset scores to per-speaker activity probabilities.
    ///
    /// Uses the "powerset to one-hot" transformation from pyannote.
    /// </summary>
    /// <param name="powersetScores">Scores for each powerset combination (frames x combinations)</param>
    /// <param name="numSpeakers">Number of speakers</param>
    /// <param name="maxSimultaneous">Maximum simultaneous speakers</param>
    /// <returns>Per-speaker activity probabilities (frames x speakers)</returns>
    public static float[][] DecodeToSpeakerActivity(float[] powersetScores, int numSpeakers, int maxSimultaneous = 2)
    {
        var combinations = GetPowersetCombinations(numSpeakers, maxSimultaneous);
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
    /// <param name="maxSimultaneous">Maximum simultaneous speakers</param>
    /// <param name="threshold">Binarization threshold</param>
    /// <returns>Array of (frame, List of active speakers)</returns>
    public static List<int>[] BinarizePowerset(
        float[] powersetScores,
        int numSpeakers,
        int maxSimultaneous = 2,
        float threshold = 0.5f)
    {
        var combinations = GetPowersetCombinations(numSpeakers, maxSimultaneous);
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
            if (bestScore >= threshold && bestCombo >= 0 && bestCombo < combinations.Length)
            {
                activeSpeakers[t] = new List<int>(combinations[bestCombo]);
            }
        }

        return activeSpeakers;
    }

    /// <summary>
    /// Count maximum simultaneous speakers from powerset combinations.
    /// </summary>
    public static int GetMaxSimultaneousSpeakers(int numSpeakers, int maxSimultaneous = 2)
    {
        var combinations = GetPowersetCombinations(numSpeakers, maxSimultaneous);
        return combinations.Max(c => c.Count);
    }

    /// <summary>
    /// Get the number of powerset combinations for N speakers.
    /// </summary>
    public static int GetNumCombinations(int numSpeakers, int maxSimultaneous = 2)
    {
        // Sum of binomial coefficients: C(N,0) + C(N,1) + ... + C(N,maxSimultaneous)
        int count = 0;
        for (int k = 0; k <= maxSimultaneous; k++)
            count += BinomialCoefficient(numSpeakers, k);
        return count;
    }

    private static int BinomialCoefficient(int n, int k)
    {
        if (k < 0 || k > n) return 0;
        if (k == 0 || k == n) return 1;
        if (k > n / 2) k = n - k;
        
        long result = 1;
        for (int i = 1; i <= k; i++)
        {
            result = result * (n - i + 1) / i;
        }
        return (int)result;
    }
}
