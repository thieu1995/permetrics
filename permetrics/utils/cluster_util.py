<<<<<<< HEAD
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist, squareform

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


def get_centroids(X: pd.DataFrame, labels):
    """
    Calculates the centroids from the data given labels

    Args:
        X (pd.DataFrame, np.ndarray): The original data that was clustered
        labels (list. pd.DataFrame, np.ndarray): The predicted cluster assignment values

    Returns:
        centroids (np.ndarray): The centroids given the input data and labels
    """
    x = pd.DataFrame(X)
    labels = np.array(labels)
    k = int(np.max(labels) + 1)
    n_cols = x.shape[1]
    centers = np.array(np.zeros(shape=(k, n_cols)))

    # Getting the centroids:
    for i in range(k):
        centers[i, :] = np.mean(x.iloc[labels == i], axis=0)

    return centers


# * Sum of Squares calculations:
def gss(X, labels):
    """
    Calculates a variety of sums of squared for future indices.

    Args:
        X (pd.DataFrame, np.ndarray): The original data that was clustered
        labels (np.array): The predicted cluster assignment values

    Returns:
        dict: The within/between group sums of squares and centroids
    """
    labels = np.array(labels)
    x = X
    centroids = get_centroids(X, labels=labels)
    allmean = np.mean(x)
    dmean = x - allmean
    allmeandist = sum(np.sum(dmean ** 2))
    centroids2 = pd.DataFrame(centroids)
    x2 = (np.array(x) - centroids2.iloc[labels, :]) ** 2
    x3 = pd.DataFrame(x2)
    # Get the sum of each row for each index
    withins = x3.sum(axis=1).reset_index().groupby(["index"]).agg({0: sum})
    withins = np.array(withins)
    wgss = float(sum(withins))
    bgss = allmeandist - wgss
    results = {"WGSS": wgss, "BGSS": bgss, "Centers": centroids}

    return results


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

    Args:
        X (pd.DataFrame, np.ndarray): The original data that was clustered
        labels (np.array): The predicted cluster assignment values


    Returns:
        dict: standard deviation, centroids, Intra-class variance, average scattering
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

    # Convert to an array for easier computation
    cluster_size = np.array(cluster_size)
    # Include an empty dimension
    cluster_size = np.expand_dims(cluster_size, axis=1)
    variance_clusters = variance_clusters / cluster_size
    variance_matrix = np.var(x, ddof=0, axis=0)

    sum_variance_clusters = []
    for u in range(k):
        sum_variance_clusters.append(
            np.sqrt(np.matmul(variance_clusters[u, ], variance_clusters[u])))

    sum_variance_clusters = np.sum(sum_variance_clusters)
    stddev = (1 / k) * np.sqrt(sum_variance_clusters)
    scatter = (1 / k) * (sum_variance_clusters / np.sqrt(np.matmul(variance_matrix, variance_matrix)))
    results = {"StdDev": stddev, "Centroids": centroids,
                "Intra-cluster Variance": variance_clusters, "Scatter": scatter}
    return results


def density_clusters(X, labels):
    """
    Used in the calculation of SDBW indices.

    Args:
        X (pd.DataFrame, np.ndarray): The original data that was clustered
        labels (np.array): The predicted cluster assignment values

    Returns:
        dict: distance and density of each cluster
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


def density_bw(self, labels):
    """
    Calculated the density between clusters.

    Args:
        X (pd.DataFrame, np.ndarray): The original data that was clustered
        labels (np.array): The predicted cluster assignment values

    Returns:
        float: The density between clusters
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
                            distance[i] = distance[i] + \
                                (x[i, j] - moy[j]) ** 2
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
        n_clusters (int): The middle point for retrieving the labels
        min_nc (int): The minimum number of clusters used in your model creation step
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
=======
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist, squareform

"Beginning of Helper Functions for later use"

def pmatch(input: list, lst: list):
    """
    A function that mimics R's pmatch function.

    Args:
    -----
    * input: the items you want
    * lst: all of the items
    
    Returns:
    --------
    * the matching indices
    """
    return [lst.index(i) for i in input]


# * Calculate the centroids from the data given labels:
# * Where input is the labels and the data:
def centers2(X: pd.DataFrame, labels):
    """
    Calculates the centroids from the data given labels

    Args:
    -----
    * X: (dataframe, np.array) The original numeric variables

    * labels: (np.array) => The labels assigned from the clustering

    Raises:
        ValueError
    """
    x = pd.DataFrame(X)
    labels = np.array(labels)
    k = int(np.max(labels) + 1)
    n_cols = x.shape[1]
    centers = np.array(np.zeros(shape=(k, n_cols)))

    # Getting the centroids:
    for i in range(k):
        centers[i, :] = np.mean(x.iloc[labels == i], axis=0)

    return centers

def autodoc_test_numpy(self, a: str, b: int = 5, c: Tuple[int, int] = (1, 2)) -> Any:
    """[summary]

    :license: MIT

    ### Parameters
    1. a: `str`
        - [description]
    2. *b : int, (default 5)
        - [description]
    3. *c : Tuple[int, int], (default (1, 2))
        - [description]

    ### Returns
    - Any
        - [description]

    Raises
    ------
    - ValueError
        - [description]

    Example:
    --------
    ```markdown
    **print(foo_bar)**
    if True:
        print(bar_foo)
    ```

    ```markdown
    :math:`\psi(r) = \exp(-2r)`
    ```
    """







# * Sum of Squares calculations:
def gss(X, labels):
    """
    Calculates a variety of Sums of Squares for future calculations

    Args:
    -----
    * X: (dataframe, np.array) The original numeric variables
    """
    labels = np.array(labels)
    x = X
    centroids = centers2(X, labels=labels)
    allmean = np.mean(x)
    dmean = x - allmean
    allmeandist = sum(np.sum(dmean ** 2))
    centroids2 = pd.DataFrame(centroids)
    x2 = (np.array(x) - centroids2.iloc[labels, :]) ** 2
    x3 = pd.DataFrame(x2)
    # Get the sum of each row for each index
    withins = x3.sum(axis=1).reset_index().groupby(["index"]).agg({0: sum})
    withins = np.array(withins)
    wgss = float(sum(withins))
    bgss = allmeandist - wgss
    results = {"WGSS": wgss, "BGSS": bgss, "Centers": centroids}

    return results

def ClustSep(X, labels):
    """
    Calculate the total separation between clusters.
    Yes, order matters in this code block.
    """
    k = int(np.max(labels) + 1)
    centroids = centers2(X, labels)
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

# Working on Average Scattering:
def AverageScattering(self, labels):
    arr_labels = np.array(labels)
    x = np.array(self.X)
    n = x.shape[0]
    num_cols = x.shape[1]
    k = int(np.max(labels) + 1)
    centroids = self.centers2(labels)
    cluster_size = pd.DataFrame(labels).value_counts().sort_index()

    variance_clusters = np.array(np.zeros(shape=(k, num_cols)))
    for u in range(k):
        for j in range(num_cols):
            for i in range(n):
                if arr_labels[i] == u:
                    variance_clusters[u, j] = variance_clusters[u,
                                                                j] + (x[i, j] - centroids[u, j]) ** 2

    # Convert to an array for easier computation
    cluster_size = np.array(cluster_size)
    # Include an empty dimension
    cluster_size = np.expand_dims(cluster_size, axis=1)
    variance_clusters = variance_clusters / cluster_size
    variance_matrix = np.var(x, ddof=0, axis=0)

    sum_variance_clusters = []
    for u in range(k):
        sum_variance_clusters.append(
            np.sqrt(np.matmul(variance_clusters[u, ], variance_clusters[u])))

    sum_variance_clusters = np.sum(sum_variance_clusters)
    stddev = (1 / k) * np.sqrt(sum_variance_clusters)
    scatter = (1 / k) * (sum_variance_clusters / np.sqrt(np.matmul(variance_matrix, variance_matrix)))
    results = {"StdDev": stddev, "Centroids": centroids,
                "Intra-cluster Variance": variance_clusters, "Scatter": scatter}
    return results

# Working on Density.Clusters for the SDBW Index:
def DensityClusters(self, labels):
    x = self.X
    x = np.array(x)
    k = int(np.max(labels)) + 1
    n = len(labels)
    num_cols = x.shape[1]
    distance = np.array(np.zeros(shape=(n, 1)))
    density = np.array(np.zeros(shape=(k, 1)))
    centroids = self.centers2(labels)
    stddev = self.AverageScattering(labels)["StdDev"]
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

def DensityBW(self, labels):
    x = self.X
    x = np.array(x)
    labels = np.array(labels)
    k = int(np.max(labels))
    n = int(x.shape[0])
    num_cols = x.shape[1]
    centroids = self.centers2(labels=labels)
    stddev = self.AverageScattering(labels)["StdDev"]
    densityBW = np.array(np.zeros(shape=(k + 1, k + 1)))
    for u in range(k + 1):
        for v in range(k + 1):
            if v != u:
                distance = np.array(np.zeros(shape=(n, 1)))
                moy = (centroids[u, :] + centroids[v, :]) / 2.0
                for i in range(n):
                    if labels[i] == u or labels[i] == v:
                        for j in range(num_cols):
                            distance[i] = distance[i] + \
                                (x[i, j] - moy[j]) ** 2
                        distance[i] = np.sqrt(distance[i])
                        if distance[i] <= stddev:
                            densityBW[u, v] += 1
    density_clust = self.DensityClusters(labels)["Density"]
    S = 0.0
    for u in range(k + 1):
        for v in range(k + 1):
            if max(density_clust[u], density_clust[v]) != 0:
                S = S + (densityBW[u, v] / max(density_clust[u], density_clust[v]))
    densityBW = S / ((k + 1) * k)
    return float(densityBW)

@staticmethod
def get_labels(labels, n_clusters: int, min_nc: int, need: str):
    """
    Parameters:
    ----------
    labels: A list of labels from the fitted models (A list of arrays).

    n_clusters: The middle point for retrieving the labels.

    min_nc: The minimum number of clusters used in your model creation step.

    need: A string representing either (K+1) labels or (K-1, K, K+1) labels
        Can be either 'clall', 'pmOne', or 'next'

    Returns:
    -------
    A Pandas dataframe with (K) labels, (K-1, K, K+1) labels or (K, K+1) labels.
    """
    if n_clusters is not None and min_nc is not None:
        if n_clusters < min_nc:
            raise ValueError(
                f"n_clusters must be greater than or equal to {min_nc}.")
    if isinstance(labels, pd.DataFrame):
        raise ValueError("Pass a **list** of labels, not a dataframe.")
    # Turns the list of labels into a neat dataframe:
    df_labels = pd.DataFrame(np.transpose(pd.DataFrame(labels)))
    # Set the column names to the number of clusters:
    df_labels.columns = list(range(min_nc, len(labels) + min_nc))

    if need in ["clall", "pmOne"]:
        # Dealing with edge cases (min. and max.):
        if n_clusters == 2:
            clall = df_labels[[n_clusters, n_clusters + 1]]
            # You'll need to create a column of 1's
            clall[0] = 1
            clall = np.array(clall)
            return clall
        # Once you reach the maximum:
        # Repeat (K) two times, but still compute (K-1) labels:
        # Idea comes from the NbClust function in the NbClust library.
        diff = n_clusters - np.max(df_labels.columns)
        if diff >= 0:
            clall = df_labels[[
                n_clusters - (diff + 1), n_clusters - diff, n_clusters - diff]]
            clall = np.array(clall)
            return clall
        else:
            # Easier to use column names instead of index, I believe.
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

>>>>>>> parent of ee8f3d2 (Moved files to same repo instead of a fork)
