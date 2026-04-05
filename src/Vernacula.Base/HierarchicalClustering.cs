namespace Vernacula.Base;

/// <summary>
/// Hierarchical agglomerative clustering (HAC) for speaker diarization.
/// 
/// This implementation mirrors pyannote.audio's clustering approach used in DiariZen.
/// The algorithm builds a dendrogram by iteratively merging the closest clusters until
/// all samples form a single cluster, then cuts the tree to obtain flat clustering.
///
/// <para><strong>Algorithm Overview:</strong></para>
/// <list type="number">
/// <item><description>Compute pairwise distances between all feature vectors</description></item>
/// <item><description>Iteratively merge the two closest clusters</description></item>
/// <item><description>Update distances using the specified linkage method</description></item>
/// <item><description>Record each merge in a linkage matrix</description></item>
/// <item><description>Cut the dendrogram at a threshold or to get N clusters</description></item>
/// </list>
///
/// <para><strong>Linkage Methods:</strong></para>
/// <list type="bullet">
/// <item><description><c>"centroid"</c>: Distance between cluster centroids (default in DiariZen)</description></item>
/// <item><description><c>"average"</c>: Average distance between all pairs (UPGMA)</description></item>
/// <item><description><c>"complete"</c>: Maximum distance between any pair (farthest neighbor)</description></item>
/// <item><description><c>"single"</c>: Minimum distance between any pair (nearest neighbor)</description></item>
/// </list>
///
/// <para><strong>DiariZen Defaults:</strong></para>
/// <list type="bullet">
/// <item><description>Method: centroid</description></item>
/// <item><description>Min cluster size: 13 frames (~0.26s at 50Hz)</description></item>
/// <item><description>Threshold: tuned via grid search (typically 0.0-2.0 for cosine distance)</description></item>
/// <item><description>Small clusters (&lt; min_cluster_size) are merged into nearest large cluster</description></item>
/// </list>
///
/// <para><strong>Linkage Matrix Format:</strong></para>
/// The returned matrix has shape (n-1) × 4, where each row represents a merge:
/// <code>[idx1, idx2, distance, num_samples]</code>
/// - idx1, idx2: Indices of clusters being merged (0 to n-1 are original samples, n+ are new clusters)
/// - distance: Distance between the merged clusters
/// - num_samples: Total samples in the new cluster
/// </para>
///
/// <para><strong>Comparison with VBx Clustering:</strong></para>
/// DiariZen also supports VBx (Variational Bayesian) clustering, which uses a two-stage approach:
/// <list type="number">
/// <item><description>AHC initialization with centroid linkage and distance threshold (0.6)</description></item>
/// <item><description>Variational Bayesian refinement using PLDA-transformed embeddings</description></item>
/// </list>
/// VBx parameters: Fa=0.07 (scales statistics), Fb=0.8 (speaker regularization), lda_dim=128
/// </para>
/// </summary>
public static class HierarchicalClustering
{
    /// <summary>
    /// Perform hierarchical agglomerative clustering.
    /// </summary>
    /// <param name="features">Feature vectors (samples x dimensions)</param>
    /// <param name="method">Linkage method: "centroid", "average", "complete", "single"</param>
    /// <returns>Linkage matrix of shape (n-1, 4): [idx1, idx2, distance, samples]</returns>
    public static double[][] Linkage(double[][] features, string method = "centroid")
    {
        int n = features.Length;
        if (n < 2)
            return Array.Empty<double[]>();

        int d = features[0].Length;
        
        // Initialize distances (pairwise Euclidean)
        // R needs to accommodate up to 2*n - 1 clusters (original n + n-1 merges)
        int maxClusters = 2 * n - 1;
        double[][] R = new double[maxClusters][];
        for (int i = 0; i < maxClusters; i++)
            R[i] = new double[maxClusters];
        
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double dist = EuclideanDistance(features[i], features[j]);
                R[i][j] = dist;
                R[j][i] = dist;
            }
        }

        // Cluster sizes and centroids
        int[] clusterSize = new int[maxClusters];
        double[]?[] clusterCentroid = new double[maxClusters][];

        for (int i = 0; i < n; i++)
        {
            clusterSize[i] = 1;
            clusterCentroid[i] = (double[])features[i].Clone();
        }

        // Linkage matrix
        double[][] linkage = new double[n - 1][];

        for (int k = 0; k < n - 1; k++)
        {
            // Find minimum distance
            double minDist = double.MaxValue;
            int iMin = -1, jMin = -1;

            for (int i = 0; i < n + k; i++)
            {
                if (clusterCentroid[i] == null) continue;
                
                for (int j = i + 1; j < n + k; j++)
                {
                    if (clusterCentroid[j] != null && R[i][j] < minDist && R[i][j] != double.MaxValue)
                    {
                        minDist = R[i][j];
                        iMin = i;
                        jMin = j;
                    }
                }
            }
            
            if (iMin == -1)
                break;

            // Record merge
            linkage[k] = new double[] { iMin, jMin, minDist, clusterSize[iMin] + clusterSize[jMin] };

            // Create new cluster
            int newIdx = n + k;
            clusterSize[newIdx] = clusterSize[iMin] + clusterSize[jMin];
            double[] leftCentroid = clusterCentroid[iMin]!;
            double[] rightCentroid = clusterCentroid[jMin]!;

            // Compute new centroid
            double[] newCentroid = new double[d];
            clusterCentroid[newIdx] = newCentroid;
            for (int dim = 0; dim < d; dim++)
            {
                newCentroid[dim] =
                    (clusterSize[iMin] * leftCentroid[dim] +
                     clusterSize[jMin] * rightCentroid[dim]) /
                    clusterSize[newIdx];
            }

            // Update distances to new cluster
            for (int i = 0; i < n + k; i++)
            {
                double[]? existingCentroid = clusterCentroid[i];
                if (i != iMin && i != jMin && existingCentroid != null)
                {
                    double dist = EuclideanDistance(existingCentroid, newCentroid);
                    R[i][newIdx] = dist;
                    R[newIdx][i] = dist;
                }
            }

            // Remove merged clusters
            R[iMin][iMin] = double.MaxValue;
            R[jMin][jMin] = double.MaxValue;
            R[iMin][newIdx] = double.MaxValue;
            R[newIdx][iMin] = double.MaxValue;
            R[jMin][newIdx] = double.MaxValue;
            R[newIdx][jMin] = double.MaxValue;
            
            // Null out centroids of merged clusters
            clusterCentroid[iMin] = null;
            clusterCentroid[jMin] = null;
            
            for (int i = 0; i < n + k; i++)
            {
                R[iMin][i] = double.MaxValue;
                R[i][iMin] = double.MaxValue;
                R[jMin][i] = double.MaxValue;
                R[i][jMin] = double.MaxValue;
            }
        }

        return linkage;
    }

    /// <summary>
    /// Cut linkage tree to get flat clustering using distance threshold.
    /// 
    /// This method cuts the dendrogram at a specified distance threshold,
    /// similar to scipy.cluster.hierarchy.fcluster with criterion="distance".
    /// All merges with distance ≤ threshold are performed, resulting in
    /// a variable number of clusters depending on the data distribution.
    ///
    /// <para><strong>Behavior:</strong></para>
    /// <list type="bullet">
    /// <item><description>Lower threshold → more clusters (fewer merges)</description></item>
    /// <item><description>Higher threshold → fewer clusters (more merges)</description></item>
    /// <item><description>Threshold=0.0 → each sample is its own cluster</description></item>
    /// <item><description>Threshold=∞ → all samples in one cluster</description></item>
    /// </list>
    ///
    /// <para><strong>Typical Values:</strong></para>
    /// For unit-normalized embeddings with cosine distance, thresholds typically
    /// range from 0.0 to 2.0. DiariZen tunes this parameter via grid search.
    /// </para>
    /// </summary>
    /// <param name="linkage">Linkage matrix from Linkage()</param>
    /// <param name="threshold">Distance threshold for cutting tree. Merges with distance ≤ threshold are performed.</param>
    /// <returns>Cluster labels for each sample (0 to k-1 where k is number of clusters)</returns>
    public static int[] FclusterThreshold(double[][] linkage, double threshold)
    {
        if (linkage.Length == 0)
            return Array.Empty<int>();

        int n = linkage.Length + 1;
        int maxClusters = 2 * n - 1;

        // Union-find (parent chain) approach — matches scipy.cluster.hierarchy.fcluster
        // with criterion="distance".  The label-replacement approach is incorrect because
        // intermediate cluster nodes (n+k) in the dendrogram never get their labels updated,
        // so later rows that reference those nodes see stale labels.
        int[] parent = new int[maxClusters];
        Array.Fill(parent, -1);

        for (int k = 0; k < linkage.Length; k++)
        {
            var row = linkage[k];
            if (row == null) continue;

            int i    = (int)row[0];
            int j    = (int)row[1];
            double dist = row[2];
            int newIdx  = n + k;

            if (dist > threshold) continue;   // cut here — don't merge

            parent[i] = newIdx;
            parent[j] = newIdx;
        }

        // For each leaf, follow the parent chain to find its root
        int[] labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            int root = i;
            while (parent[root] != -1)
                root = parent[root];
            labels[i] = root;
        }

        var rootToLabel = new System.Collections.Generic.Dictionary<int, int>();
        int nextLabel = 0;
        for (int i = 0; i < n; i++)
        {
            int root = labels[i];
            if (!rootToLabel.ContainsKey(root))
                rootToLabel[root] = nextLabel++;
            labels[i] = rootToLabel[root];
        }

        return labels;
    }

    /// <summary>
    /// Cut linkage tree to get exactly t clusters.
    /// 
    /// This method performs exactly (n - t) merges from the linkage matrix,
    /// similar to scipy.cluster.hierarchy.fcluster with criterion="maxclust".
    /// This guarantees exactly t clusters regardless of distance thresholds.
    ///
    /// <para><strong>Usage:</strong></para>
    /// Use this method when you need a specific number of clusters, such as
    /// when you have prior knowledge about the expected number of speakers.
    /// In DiariZen, this is used with minSpeakers to avoid over-segmentation.
    /// </para>
    ///
    /// <para><strong>Note:</strong></para>
    /// If t ≥ n (number of samples), each sample becomes its own cluster.
    /// If t = 1, all samples are merged into a single cluster.
    /// </para>
    /// </summary>
    /// <param name="linkage">Linkage matrix from Linkage()</param>
    /// <param name="t">Target number of clusters (must be ≥ 1 and ≤ number of samples)</param>
    /// <returns>Cluster labels for each sample (0 to t-1)</returns>
    public static int[] FclusterMaxClust(double[][] linkage, int t)
    {
        if (linkage.Length == 0)
            return Array.Empty<int>();

        int n = linkage.Length + 1;
        int maxClusters = 2 * n - 1;
        int[] parent = new int[maxClusters];
        Array.Fill(parent, -1);
        
        int mergesToPerform = n - t;
        
        for (int k = 0; k < Math.Min(linkage.Length, mergesToPerform); k++)
        {
            int i = (int)linkage[k][0];
            int j = (int)linkage[k][1];
            int newCluster = n + k;
            
            parent[i] = newCluster;
            parent[j] = newCluster;
        }
        
        int[] labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            int root = i;
            while (parent[root] != -1)
                root = parent[root];
            labels[i] = root;
        }
        
        var rootToLabel = new System.Collections.Generic.Dictionary<int, int>();
        int nextLabel = 0;
        for (int i = 0; i < n; i++)
        {
            int root = labels[i];
            if (!rootToLabel.ContainsKey(root))
                rootToLabel[root] = nextLabel++;
            labels[i] = rootToLabel[root];
        }
        
        return labels;
    }

    private static double EuclideanDistance(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }
}
