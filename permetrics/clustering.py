<<<<<<< HEAD
# Created by "Thieu" at 18:07, 18/07/2020 ----------%
#       Email: miq_qedquinn@yahoo.com               %
#       Github: https://github.com/N3uralN3twork    %
# --------------------------------------------------%

import numpy as np
import pandas as pd
from itertools import chain
from collections import Counter
from scipy.stats import f
from sklearn.metrics.pairwise import pairwise_distances_chunked
from sklearn import metrics
from scipy.spatial.distance import pdist, cdist, squareform
import functools

from utils.cluster_util import average_scattering, cluster_sep, density_bw, density_clusters, get_centroids, get_labels, gss, pmatch


class InternalMetric(object):
    """
    This class contains a variety of clustering metrics (internal only)

    Notes
    ~~~~~
    + An extension of scikit-learn metrics section, with the addition of many more internal metrics.
    + https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
    """

    def __init__(self, data, X, y, decimal=5):
        """
        Args:
            * X (pd.DataFrame, np.ndarray): The original data that was clustered
            * y_pred (list. pd.DataFrame, np.ndarray): The predicted cluster assignment values
            * decimal (int): The number of fractional parts after the decimal point (Optional, default=5)
        """
        self.data = data
        self.X = X
        self.y = y
        self.distances = pdist(self.X, metric="euclidean")

    def ball_hall_index(self, labels, n_clusters: int, min_nc: int):
        """
        The Ball-Hall Index (1995) is the mean of the mean dispersion across all clusters.

        The **largest difference** between successive clustering levels indicates the optimal number of clusters.

        Args:

        labels: The predicted labels for the given clustering technique

        Returns:
            BH (float): The Ball-Hall index
        """

        use_labels = get_labels(labels, n_clusters, min_nc, need="Single")
        use_labels = np.array(use_labels).flatten()
        wgss = []
        n_labels = len(set(use_labels))

        for k in range(n_labels):
            cluster_k = self.X[use_labels == k]
            mean_k = np.mean(cluster_k, axis=0)
            wgss.append(np.sum((cluster_k - mean_k) ** 2))

        BH = np.sum(wgss) / (n_labels)

        return BH



=======
# Created by "Thieu" at 18:07, 18/07/2020 ----------%
#       Email: miq_qedquinn@yahoo.com               %
#       Github: https://github.com/N3uralN3twork    %
# --------------------------------------------------%

import numpy as np
import pandas as pd
from itertools import chain
from collections import Counter
import re
from scipy.stats import f
from sklearn.metrics.pairwise import pairwise_distances_chunked
from sklearn import metrics
from scipy.spatial.distance import pdist, cdist, squareform
import functools


class ClusteringMetric(object):
    """
    This class contains a variety of clustering metrics (internal only)

    Notes
    ~~~~~
    + An extension of scikit-learn metrics section, with the addition of many more internal metrics.
    + https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
    """

    def __init__(self, data, X, y, decimal=5, **kwargs):
        """
        Args:
        ----
            * X (tuple, list, np.ndarray):
                - The ground truth values
            * y (tuple, list, np.ndarray): 
                - The prediction values
            * decimal (int): 
                - The number of fractional parts after the decimal point
            * **kwargs ():
        """

        self.data = data
        self.X = X
        self.y = y
        self.distances = pdist(self.X, metric="euclidean")

        if kwargs is None: kwargs = {}
        self.one_dim = False
    



>>>>>>> parent of ee8f3d2 (Moved files to same repo instead of a fork)
