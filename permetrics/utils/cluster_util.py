import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform

"Beginning of Helper Functions for later use"

def pmatch(input: list, lst: list):
    """
    A function that mimics R's pmatch function

    Args:
        input (list): The input list
        lst (list): The list of interest

    Returns:
        list: A list of integers representing indices
    """
    return [lst.index(i) for i in input]


def get_centroids(X, labels):
    """
    Calculates the centroids from the data given, for each class.

    Args:
        X (pd.DataFrame, np.ndarray): The original data that was clustered
        labels (list, np.ndarray): The predicted cluster assignment values

    Returns:
        centroids (np.ndarray): The centroids given the input data and labels
    """
    n_samples, n_features = X.shape
    n_classes = len(np.unique(labels))
    # * Mask mapping each class to its members.
    centroids = np.empty((n_classes, n_features), dtype=np.float64)
    # * Number of clusters in each class.
    nk = np.zeros(n_classes)

    for k in range(n_classes):
        centroid_mask = labels == k
        nk[k] = np.sum(centroid_mask)
        centroids[k] = X[centroid_mask].mean(axis=0)
    
    return centroids


# * Sum of Squares calculations:
def general_sums_of_squares(X, labels):
    """
    Calculates a variety of sums of squares for a given index.

    Args:
        X (pd.DataFrame, np.ndarray): The original data that was clustered
        labels (np.array): The predicted cluster assignment values

    Returns:
        dict: The within/between group sums of squares and centroids
    """
    labels = np.array(labels)
    n_classes = len(set(labels))
    BGSS, WGSS = 0.0, 0.0
    mean = np.mean(X, axis=0)

    for k in range(n_classes):
        cluster_k = X[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        BGSS += len(cluster_k) * np.sum((mean_k - mean) ** 2)
        WGSS += np.sum((cluster_k - mean_k) ** 2)

    return {"BGSS": BGSS, "WGSS": WGSS}


def cluster_sep(X, labels):
    """
    Calculate the total separation between clusters.

    Args:
        X (pd.DataFrame, np.ndarray): The original data that was clustered
        labels (np.array): The predicted cluster assignment values

    Returns:
        list: The total distance/separation between each cluster
    """
    k = int(np.max(labels) + 1)
    centroids = get_centroids(X, labels)
    dist_centers = squareform(pdist(centroids))
    num_cols = dist_centers.shape[1]
    Dmin = np.min(pdist(centroids))
    Dmax = np.max(pdist(centroids))
    s2 = 0
    for u in range(k):
        s1 = 0
        for j in range(num_cols):
            s1 = s1 + dist_centers[u, j]
        s2 = s2 + (1 / s1)
    Dis = (Dmax / Dmin) * s2

    return Dis


def average_scattering(X, labels):
    """
    Calculates the average scattering for a given set of clusters.

    This can also be thought of as a vector of variances for a particular cluster.

    Args:
        X (pd.DataFrame, np.ndarray): The original data that was clustered
        labels (np.array): The predicted cluster assignment values

    Returns:
        results (dict): standard deviation, centroids, Intra-class variance, average scattering
    """
    arr_labels = np.array(labels)
    x = np.array(X)
    n = x.shape[0]
    num_cols = x.shape[1]
    k = int(np.max(labels) + 1)
    centroids = get_centroids(X, labels)
    cluster_size = pd.DataFrame(labels).value_counts().sort_index()

    variance_clusters = np.array(np.zeros(shape=(k, num_cols)))
    for u in range(k):
        for j in range(num_cols):
            for i in range(n):
                if arr_labels[i] == u:
                    variance_clusters[u, j] = variance_clusters[u, j] + (x[i, j] - centroids[u, j]) ** 2

    # * Convert to an array for easier computation
    cluster_size = np.array(cluster_size)
    # * Include an empty dimension
    cluster_size = np.expand_dims(cluster_size, axis=1)
    variance_clusters = variance_clusters / cluster_size
    variance_matrix = np.var(x, ddof=0, axis=0)

    sum_variance_clusters = []
    for u in range(k):
        sum_variance_clusters.append(np.sqrt(np.matmul(variance_clusters[u, ], variance_clusters[u])))

    sum_variance_clusters = np.sum(sum_variance_clusters)
    stddev = (1 / k) * np.sqrt(sum_variance_clusters)
    scatter = (1 / k) * (sum_variance_clusters / np.sqrt(np.matmul(variance_matrix, variance_matrix)))
    results = {"StdDev": stddev
            , "Centroids": centroids
            , "Intra-cluster Variance": variance_clusters
            , "Scatter": scatter}
    
    return results


def density_clusters(X, labels):
    """
    Used in the calculation of SDBW indices.

    Args:
        X (pd.DataFrame, np.ndarray): The original data that was clustered
        labels (np.array): The predicted cluster assignment values

    Returns:
        results (dict): distance and density of each cluster
    """
    
    x = X
    x = np.array(x)
    k = int(np.max(labels)) + 1
    n = len(labels)
    num_cols = x.shape[1]
    distance = np.array(np.zeros(shape=(n, 1)))
    density = np.array(np.zeros(shape=(k, 1)))
    centroids = get_centroids(X, labels)
    stddev = average_scattering(X, labels)["StdDev"]
    for i in range(n):
        u = 0
        while labels[i] != u:
            u += 1
        for j in range(num_cols):
            distance[i] = distance[i] + (x[i, j] - centroids[u, j]) ** 2
        distance[i] = np.sqrt(distance[i])
        if distance[i] <= stddev:
            density[u] += 1
    results = {"Distance": distance, "Density": density}
    return results


def density_between(self, labels):
    """
    Calculates the density between clusters, aka inter-cluster density.

    Args:
        X (pd.DataFrame, np.ndarray): The original data that was clustered
        labels (np.array): The predicted cluster assignment values

    Returns:
        density_bw (float): The density between clusters
    """
    x = self.X
    x = np.array(x)
    labels = np.array(labels)
    k = int(np.max(labels))
    n = int(x.shape[0])
    num_cols = x.shape[1]
    centroids = self.get_centroids(labels=labels)
    stddev = self.average_scattering(labels)["StdDev"]
    density_bw = np.array(np.zeros(shape=(k + 1, k + 1)))
    for u in range(k + 1):
        for v in range(k + 1):
            if v != u:
                distance = np.array(np.zeros(shape=(n, 1)))
                moy = (centroids[u, :] + centroids[v, :]) / 2.0
                for i in range(n):
                    if labels[i] == u or labels[i] == v:
                        for j in range(num_cols):
                            distance[i] = distance[i] + (x[i, j] - moy[j]) ** 2
                        distance[i] = np.sqrt(distance[i])
                        if distance[i] <= stddev:
                            density_bw[u, v] += 1
    density_clust = self.density_clusters(labels)["Density"]
    S = 0.0
    for u in range(k + 1):
        for v in range(k + 1):
            if max(density_clust[u], density_clust[v]) != 0:
                S = S + (density_bw[u, v] / max(density_clust[u], density_clust[v]))
    density_bw = S / ((k + 1) * k)
    return float(density_bw)


def get_labels(labels, n_clusters: int, min_nc: int, need: str):
    """
    A helper function to return the correct number of labels for a particular metric.

    Args:
        labels (list): A list of labels from the fittend models
        n_clusters (int): The median number of clusters to get indices for
        min_nc (int): The minimum number of clusters to get indices for
        need (str): A string representing either (K+1) labels or (K-1, K, K+1) labels
            Can be either `clall`, `pmOne`, or `next`
    
    Returns:
        A dataframe with (K) labels, (K-1, K, K+1) labels or (K, K+1) labels.
    """
    if n_clusters is not None and min_nc is not None:
        if n_clusters < min_nc:
            raise ValueError(
                f"n_clusters must be greater than or equal to {min_nc}.")
    if isinstance(labels, pd.DataFrame):
        raise ValueError("Pass a **list** of labels, not a dataframe.")
    # * Turns the list of labels into a neat dataframe:
    df_labels = pd.DataFrame(np.transpose(pd.DataFrame(labels)))
    # * Set the column names to the number of clusters:
    df_labels.columns = list(range(min_nc, len(labels) + min_nc))

    if need in ["clall", "pmOne"]:
        # * Dealing with edge cases (min. and max.):
        if n_clusters == 2:
            clall = df_labels[[n_clusters, n_clusters + 1]]
            # * You'll need to create a column of 1's
            clall[0] = 1
            clall = np.array(clall)
            return clall
        # * Once you reach the maximum:
        # * Repeat (K) two times, but still compute (K-1) labels:
        # * Idea comes from the NbClust function in the NbClust library.
        diff = n_clusters - np.max(df_labels.columns)
        if diff >= 0:
            clall = df_labels[[
                n_clusters - (diff + 1), n_clusters - diff, n_clusters - diff]]
            clall = np.array(clall)
            return clall
        else:
            # * Easier to use column names instead of index, I believe.
            clall = df_labels[[n_clusters - 1, n_clusters, n_clusters + 1]]
            clall = np.array(clall)
            return clall
    elif need == "next":
        if n_clusters + 1 > len(labels) + min_nc:
            raise ValueError(f"STOP: You are trying to access labels in the {n_clusters + 1}th position,"
                                f" which don't exist. Maximum label position is {max(df_labels.columns)}")
        diff = n_clusters - np.max(df_labels.columns)
        if diff >= 0:
            clall = df_labels[[
                n_clusters - (diff + 1), n_clusters - diff, n_clusters - diff]]
            clall = np.array(clall)
            return clall
        else:
            return np.array(df_labels[[n_clusters, n_clusters + 1]])
    elif need in ["Regular", "Normal", "Single", "single", "normal", "regular"]:
        return np.array(df_labels[[n_clusters]])


