namespace Vernacula.Base.Inference;

/// <summary>
/// Per-backend VRAM cost model consulted by <see cref="BatchSizer.Plan"/>.
/// </summary>
/// <remarks>
/// The abstraction keeps the model-specific architecture constants (layer
/// count, head dim, encoder downsampling, etc.) in the backend's own file:
/// <see cref="BatchSizer"/> only asks for a single peak-bytes number per
/// trial batch and lets the backend decide which term dominates (KV cache,
/// encoder activation spike, both via Math.Max, …).
/// </remarks>
public interface IBatchCostModel
{
    /// <summary>
    /// Worst-case peak VRAM for a prospective batch of <paramref name="batchSize"/>
    /// segments whose longest member is <paramref name="maxDurationSec"/> seconds.
    /// Return the maximum of every transient allocation the batch will produce
    /// (KV cache, encoder activations, mel buffers, …) — <see cref="BatchSizer"/>
    /// compares it against the VRAM budget as a single scalar.
    /// </summary>
    long EstimatePeakBytes(int batchSize, double maxDurationSec);
}

/// <summary>
/// One contiguous batch produced by <see cref="BatchSizer.Plan"/>: the original
/// segment indices (i.e. positions in the caller's input list) that the backend
/// should stack into this forward pass. Indices within a batch are sorted by
/// ascending duration so the worst-case segment is always last.
/// </summary>
public readonly record struct Batch(int[] SegmentIndices)
{
    public int Count => SegmentIndices.Length;
}

/// <summary>
/// VRAM-aware dynamic batch planner. Sorts segments by ascending duration,
/// then greedily grows each batch one segment at a time, consulting the
/// caller-supplied <see cref="IBatchCostModel"/> until either the budget is
/// exceeded or a hard cap is hit.
/// </summary>
/// <remarks>
/// Extracted verbatim from Cohere's <c>RecognizeBatched</c> loop. The
/// ascending sort is load-bearing: stragglers (shorter segments that have
/// emitted EOS) keep stepping until the longest member finishes, so batches
/// of similar length minimise wasted decoder steps.
///
/// Forward-progress guarantee: the first segment of every batch is always
/// admitted even if it alone would breach the budget. This preserves the
/// original Cohere behaviour where pathologically large single segments do
/// not hang the planner.
/// </remarks>
public static class BatchSizer
{
    public static IReadOnlyList<Batch> Plan(
        IReadOnlyList<double> durationsSec,
        IBatchCostModel costs,
        long vramBudgetBytes,
        int maxBatchSize)
    {
        if (durationsSec.Count == 0) return [];
        if (maxBatchSize <= 0) throw new ArgumentOutOfRangeException(nameof(maxBatchSize));

        int[] order = new int[durationsSec.Count];
        for (int i = 0; i < order.Length; i++) order[i] = i;
        Array.Sort(order, (a, b) => durationsSec[a].CompareTo(durationsSec[b]));

        var batches = new List<Batch>();
        int pos = 0;
        while (pos < order.Length)
        {
            int batchSize = 0;
            double maxDur = 0;

            while (pos + batchSize < order.Length && batchSize < maxBatchSize)
            {
                double candDur = durationsSec[order[pos + batchSize]];
                double newMax = Math.Max(maxDur, candDur);
                long peak = costs.EstimatePeakBytes(batchSize + 1, newMax);
                if (batchSize > 0 && peak > vramBudgetBytes)
                    break;
                maxDur = newMax;
                batchSize++;
            }

            int[] indices = new int[batchSize];
            Array.Copy(order, pos, indices, 0, batchSize);
            batches.Add(new Batch(indices));
            pos += batchSize;
        }

        return batches;
    }
}
