namespace Parakeet.Base;

/// <summary>
/// Agglomerative Hierarchical Clustering (AHC) implementation for speaker diarization.
/// 
/// This is a pure C# implementation of centroid-based hierarchical clustering,
/// suitable for porting from SciPy's linkage/fcluster functions.
/// </summary>
public static class HierarchicalClustering
{
    /// <summary>
    /// Perform agglomerative hierarchical clustering using centroid linkage.
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
        double[][] R = new double[n][];
        for (int i = 0; i < n; i++)
            R[i] = new double[n];
        
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
        int[] clusterSize = new int[2 * n - 1];
        double[][] clusterCentroid = new double[2 * n - 1][];
        
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
                for (int j = i + 1; j < n + k; j++)
                {
                    if (R[i][j] < minDist)
                    {
                        minDist = R[i][j];
                        iMin = i;
                        jMin = j;
                    }
                }
            }

            // Record merge
            linkage[k] = new double[] { iMin, jMin, minDist, clusterSize[iMin] + clusterSize[jMin] };
            
            // Create new cluster
            int newIdx = n + k;
            clusterSize[newIdx] = clusterSize[iMin] + clusterSize[jMin];
            
            // Compute new centroid
            clusterCentroid[newIdx] = new double[d];
            for (int dim = 0; dim < d; dim++)
            {
                clusterCentroid[newIdx][dim] = 
                    (clusterSize[iMin] * clusterCentroid[iMin][dim] + 
                     clusterSize[jMin] * clusterCentroid[jMin][dim]) / 
                    clusterSize[newIdx];
            }

            // Update distances to new cluster
            for (int i = 0; i < n + k; i++)
            {
                if (i != iMin && i != jMin)
                {
                    double dist = EuclideanDistance(clusterCentroid[i], clusterCentroid[newIdx]);
                    R[i][newIdx] = dist;
                    R[newIdx][i] = dist;
                }
            }

            // Remove merged clusters
            R[iMin][iMin] = double.MaxValue;
            R[jMin][jMin] = double.MaxValue;
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
    /// Cut linkage tree to get flat clustering.
    /// </summary>
    /// <param name="linkage">Linkage matrix from Linkage()</param>
    /// <param name="threshold">Distance threshold for cutting tree</param>
    /// <returns>Cluster labels for each sample</returns>
    public static int[] FclusterThreshold(double[][] linkage, double threshold)
    {
        if (linkage.Length == 0)
            return Array.Empty<int>();

        int n = linkage.Length + 1;
        int[] labels = new int[n];
        for (int i = 0; i < n; i++)
            labels[i] = i; // Each sample starts in its own cluster

        // Process merges in order
        int nextLabel = n;
        foreach (var row in linkage)
        {
            int i = (int)row[0];
            int j = (int)row[1];
            double dist = row[2];

            if (dist > threshold)
                break;

            // Merge clusters
            int labelI = labels[i];
            int labelJ = labels[j];
            
            if (labelI != labelJ)
            {
                // Reassign all samples in cluster J to cluster I
                for (int k = 0; k < n; k++)
                {
                    if (labels[k] == labelJ)
                        labels[k] = labelI;
                }
            }
        }

        // Relabel to consecutive integers starting from 0
        var labelMap = new System.Collections.Generic.Dictionary<int, int>();
        int newLabel = 0;
        for (int i = 0; i < n; i++)
        {
            if (!labelMap.ContainsKey(labels[i]))
                labelMap[labels[i]] = newLabel++;
            labels[i] = labelMap[labels[i]];
        }

        return labels;
    }

    /// <summary>
    /// Cut linkage tree to get exactly t clusters.
    /// </summary>
    public static int[] FclusterMaxClust(double[][] linkage, int t)
    {
        if (linkage.Length == 0)
            return Array.Empty<int>();

        int n = linkage.Length + 1;
        
        // Start with each sample in its own cluster
        var clusters = new System.Collections.Generic.List<System.Collections.Generic.List<int>>();
        for (int i = 0; i < n; i++)
            clusters.Add(new System.Collections.Generic.List<int> { i });

        // Process merges until we have t clusters
        int merges = 0;
        foreach (var row in linkage)
        {
            if (clusters.Count <= t)
                break;

            int i = (int)row[0];
            int j = (int)row[1];
            
            // Find which clusters contain i and j
            int clusterI = -1, clusterJ = -1;
            for (int c = 0; c < clusters.Count; c++)
            {
                if (clusters[c].Contains(i)) clusterI = c;
                if (clusters[c].Contains(j)) clusterJ = c;
            }

            if (clusterI >= 0 && clusterJ >= 0 && clusterI != clusterJ)
            {
                // Merge clusters
                clusters[clusterI].AddRange(clusters[clusterJ]);
                clusters.RemoveAt(clusterJ);
            }
        }

        // Assign labels
        int[] labels = new int[n];
        for (int c = 0; c < clusters.Count; c++)
        {
            foreach (int idx in clusters[c])
                labels[idx] = c;
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
