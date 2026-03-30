namespace Parakeet.Base;

/// <summary>
/// Variational Bayes x-vector clustering (VBx) for speaker diarization.
///
/// <para>Implements the GMM-VBx variant (loopProb=0) from:</para>
/// <para>Landini et al., "Bayesian HMM clustering of x-vector sequences (VBx)
/// in speaker diarization: theory, implementation and analysis on standard tasks"</para>
///
/// <para><strong>Inputs:</strong></para>
/// <list type="bullet">
/// <item>fea      — (N, D) L2-normalised embeddings in PLDA space</item>
/// <item>phi      — (D,) PLDA diagonal eigenvalues (between-class covariance)</item>
/// <item>initLabels — (N,) initial cluster assignments from AHC</item>
/// <item>Fa, Fb   — VBx hyperparameters (default 0.07, 0.8)</item>
/// </list>
/// </summary>
public static class VbxClustering
{
    private const double Eps     = 1e-8;
    private const double Epsilon = 1e-4;  // convergence tolerance

    /// <summary>
    /// Run VBx clustering and return hard speaker assignments (0-based).
    /// </summary>
    /// <param name="fea">N × D feature matrix (rows are L2-normalised embeddings).</param>
    /// <param name="phi">D-dimensional PLDA eigenvalue vector.</param>
    /// <param name="initLabels">N-dimensional initial cluster labels (from AHC).</param>
    /// <param name="fa">Sufficient statistics scale factor.</param>
    /// <param name="fb">Speaker regularisation coefficient.</param>
    /// <param name="maxIters">Maximum VB iterations.</param>
    /// <returns>Hard assignment labels, one per embedding.</returns>
    public static int[] Cluster(
        double[][] fea,
        double[]   phi,
        int[]      initLabels,
        double fa      = 0.07,
        double fb      = 0.8,
        int    maxIters = 20)
    {
        int n = fea.Length;
        int d = phi.Length;
        int k = initLabels.Max() + 1;

        if (n == 0 || k == 0) return initLabels;
        if (k == 1)           return initLabels;

        // ── Initialise gamma (N×K soft assignments) from hard AHC labels ─
        var gamma = new double[n, k];
        const double smoothing = 7.0;
        for (int i = 0; i < n; i++)
        {
            int lbl = initLabels[i];
            if (lbl < 0 || lbl >= k) continue;
            for (int j = 0; j < k; j++)
                gamma[i, j] = smoothing / (k - 1 + smoothing);
            gamma[i, lbl] += 1.0 - smoothing / (k - 1 + smoothing) * k;
            if (gamma[i, lbl] < 0) gamma[i, lbl] = 0;
            // Re-normalise row
            double rowSum = 0;
            for (int j = 0; j < k; j++) rowSum += gamma[i, j];
            if (rowSum > 0) for (int j = 0; j < k; j++) gamma[i, j] /= rowSum;
        }

        // ── Speaker priors ────────────────────────────────────────────────
        var pi = new double[k];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++) pi[j] += gamma[i, j];
        double piSum = pi.Sum();
        for (int j = 0; j < k; j++) pi[j] /= piSum;

        // ── Pre-compute constants ─────────────────────────────────────────
        // G[i] = -0.5 * (||x_i||² + D*ln(2π))   (per-frame log-const, shape N)
        var g = new double[n];
        double logTwoPiD = d * Math.Log(2 * Math.PI);
        for (int i = 0; i < n; i++)
        {
            double sq = 0;
            for (int dim = 0; dim < d; dim++) sq += fea[i][dim] * fea[i][dim];
            g[i] = -0.5 * (sq + logTwoPiD);
        }

        // V = sqrt(phi)  — element-wise
        var v = new double[d];
        for (int dim = 0; dim < d; dim++) v[dim] = Math.Sqrt(phi[dim]);

        // rho = fea * v   (N × D)
        var rho = new double[n][];
        for (int i = 0; i < n; i++)
        {
            rho[i] = new double[d];
            for (int dim = 0; dim < d; dim++) rho[i][dim] = fea[i][dim] * v[dim];
        }

        // ── VBx iterations ────────────────────────────────────────────────
        double prevElbo = double.NegativeInfinity;

        var invL  = new double[k, d];   // (K × D)
        var alpha = new double[k, d];   // (K × D)
        var logP  = new double[n, k];   // (N × K) log-likelihoods

        for (int iter = 0; iter < maxIters; iter++)
        {
            // ── Speaker model update: invL, alpha ─────────────────────────
            // gammaSumPerSpk[j] = Σ_i gamma[i,j]
            var gammaSumPerSpk = new double[k];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < k; j++) gammaSumPerSpk[j] += gamma[i, j];

            // invL[j, dim] = 1 / (1 + (Fa/Fb) * gammaSumPerSpk[j] * phi[dim])
            double faOverFb = fa / fb;
            for (int j = 0; j < k; j++)
                for (int dim = 0; dim < d; dim++)
                    invL[j, dim] = 1.0 / (1.0 + faOverFb * gammaSumPerSpk[j] * phi[dim]);

            // alpha[j, dim] = (Fa/Fb) * invL[j,dim] * Σ_i gamma[i,j] * rho[i,dim]
            for (int j = 0; j < k; j++)
                for (int dim = 0; dim < d; dim++) alpha[j, dim] = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < k; j++)
                {
                    double g_ij = gamma[i, j];
                    for (int dim = 0; dim < d; dim++)
                        alpha[j, dim] += g_ij * rho[i][dim];
                }
            for (int j = 0; j < k; j++)
                for (int dim = 0; dim < d; dim++)
                    alpha[j, dim] *= faOverFb * invL[j, dim];

            // ── Per-frame log-likelihoods ─────────────────────────────────
            // logP[i,j] = Fa * (rho[i] · alpha[j] - 0.5*(invL[j]+alpha[j]²)·phi + G[i])
            for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
            {
                double dot = 0, reg = 0;
                for (int dim = 0; dim < d; dim++)
                {
                    dot += rho[i][dim] * alpha[j, dim];
                    reg += (invL[j, dim] + alpha[j, dim] * alpha[j, dim]) * phi[dim];
                }
                logP[i, j] = fa * (dot - 0.5 * reg + g[i]);
            }

            // ── Responsibilities (GMM update, loopProb=0) ─────────────────
            double elbo = 0;
            var lpi = new double[k];
            for (int j = 0; j < k; j++) lpi[j] = Math.Log(pi[j] + Eps);

            for (int i = 0; i < n; i++)
            {
                // log-sum-exp for numerical stability
                double maxLp = double.NegativeInfinity;
                for (int j = 0; j < k; j++)
                {
                    double lp = logP[i, j] + lpi[j];
                    if (lp > maxLp) maxLp = lp;
                }
                double sumExp = 0;
                for (int j = 0; j < k; j++) sumExp += Math.Exp(logP[i, j] + lpi[j] - maxLp);
                double logPx = maxLp + Math.Log(sumExp);
                elbo += logPx;

                for (int j = 0; j < k; j++)
                    gamma[i, j] = Math.Exp(logP[i, j] + lpi[j] - logPx);
            }

            // ELBO speaker regularisation term
            for (int j = 0; j < k; j++)
                for (int dim = 0; dim < d; dim++)
                    elbo += fb * 0.5 * (Math.Log(invL[j, dim]) - invL[j, dim] - alpha[j, dim] * alpha[j, dim] + 1.0);

            // ── Update priors ─────────────────────────────────────────────
            double piTotal = 0;
            for (int j = 0; j < k; j++) { pi[j] = 0; for (int i = 0; i < n; i++) pi[j] += gamma[i, j]; piTotal += pi[j]; }
            for (int j = 0; j < k; j++) pi[j] /= piTotal;

            if (iter > 0 && elbo - prevElbo < Epsilon) break;
            prevElbo = elbo;
        }

        // ── Hard assignment: argmax per row ───────────────────────────────
        var labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            int bestJ = 0;
            double bestG = gamma[i, 0];
            for (int j = 1; j < k; j++)
                if (gamma[i, j] > bestG) { bestG = gamma[i, j]; bestJ = j; }
            labels[i] = bestJ;
        }

        // Compact: drop empty clusters, renumber 0..K'-1
        var used = new HashSet<int>(labels);
        var remap = new Dictionary<int, int>();
        int next = 0;
        foreach (int lbl in Enumerable.Range(0, k).Where(j => used.Contains(j)))
            remap[lbl] = next++;
        for (int i = 0; i < n; i++) labels[i] = remap[labels[i]];

        return labels;
    }
}
