# Created by "Matt Q." at 23:05, 27/10/2022 --------%
#       Github: https://github.com/N3uralN3twork    %
# --------------------------------------------------%

import functools
from collections import Counter
from itertools import chain

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import f

from utils.cluster_util import (average_scattering, cluster_sep,
                                density_between, density_clusters,
                                general_sums_of_squares, get_centroids,
                                get_labels, pmatch)


class InternalMetric(object):
    """
    This class contains a variety of clustering metrics (internal only)

    Internal clustering metrics only consider the data that used for clustering, disregarding any additional labels.

    External clustering metrics utilize both the data used for clustering as well as a set of true clustering solutions.

    Notes
    ~~~~~
    + An extension of scikit-learn metrics section, with the addition of many more internal metrics.
    + https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
    """

    def __init__(self, data, X, y, decimal=5):
        """
        Args:
            X (pd.DataFrame, np.ndarray): The original data that was clustered
            y_pred (list. pd.DataFrame, np.ndarray): The predicted cluster assignment values
            decimal (int): The number of fractional parts after the decimal point (Optional, default=5)
        """
        self.data = data
        self.X = X
        self.y = y
        self.distances = pdist(self.X, metric="euclidean")
        self.decimal = decimal


    def ball_hall_index(self, labels, n_clusters: int, min_nc: int):
        """
        The Ball-Hall Index (1995) is the mean of the mean dispersion across all clusters.

        The **largest difference** between successive clustering levels indicates the optimal number of clusters.

        Args:
            labels (list, pd.DataFrame, np.ndarray): The predicted cluster assignment values
            n_clusters (int): The requested/median number of clusters to retrieve indices for
            min_nc (int): The minimum number of clusters to retrieve indices for

        Returns:
            BH (float): The Ball-Hall index
        """

        # ! use_labels = get_labels(labels, n_clusters, min_nc, need="single")
        use_labels = labels
        use_labels = np.array(use_labels).flatten()
        wgss = []
        n_classes = len(set(use_labels))

        # * For each cluster, find the centroid and then the within-group SSE
        for k in range(n_classes):
            centroid_mask = use_labels == k
            cluster_k = self.X[centroid_mask]
            centroid = np.mean(cluster_k, axis=0)
            wgss.append(np.sum((cluster_k - centroid)**2))

        BH = np.sum(wgss) / (n_classes)

        return BH


