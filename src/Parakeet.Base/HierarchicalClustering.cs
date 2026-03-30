namespace Parakeet.Base;

/// <summary>
/// Hierarchical agglomerative clustering implementation.
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
        double[][] clusterCentroid = new double[maxClusters][];

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
                if (i != iMin && i != jMin && clusterCentroid[i] != null)
                {
                    double dist = EuclideanDistance(clusterCentroid[i], clusterCentroid[newIdx]);
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
        int maxClusters = 2 * n - 1;
        int[] labels = new int[maxClusters];
        for (int i = 0; i < maxClusters; i++)
            labels[i] = i;

        foreach (var row in linkage)
        {
            int i = (int)row[0];
            int j = (int)row[1];
            double dist = row[2];

            if (dist > threshold)
                break;

            int labelI = labels[i];
            int labelJ = labels[j];
            
            if (labelI != labelJ)
            {
                for (int k = 0; k < maxClusters; k++)
                {
                    if (labels[k] == labelJ)
                        labels[k] = labelI;
                }
            }
        }

        var labelMap = new System.Collections.Generic.Dictionary<int, int>();
        int newLabel = 0;
        for (int i = 0; i < n; i++)
        {
            if (!labelMap.ContainsKey(labels[i]))
                labelMap[labels[i]] = newLabel++;
            labels[i] = labelMap[labels[i]];
        }

        var result = new int[n];
        Array.Copy(labels, result, n);
        return result;
    }

    /// <summary>
    /// Cut linkage tree to get exactly t clusters.
    /// </summary>
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
