# !/usr/bin/env python
# Created by "Matt Q." at 23:05, 27/10/2022 --------%
#       Github: https://github.com/N3uralN3twork    %
#                                                   %
# Improved by: "Thieu" at 11:45, 25/07/2023 --------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import functools
from collections import Counter
from itertools import chain
import numpy as np
from permetrics.evaluator import Evaluator
from permetrics.utils.data_util import *
from permetrics.utils.cluster_util import (get_min_dist, get_centroids, general_sums_of_squares, pdist,
                                cdist, squareform, get_labels,
                                average_scattering, cluster_sep, density_between, density_clusters, pmatch)


class ClusteringMetric(Evaluator):
    """
    This is class contains all clustering metrics (for both internal and external performance metrics)

    Notes
    ~~~~~
    + An extension of scikit-learn metrics section, with the addition of many more internal metrics.
    + https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
    """

    def __init__(self, y_true=None, y_pred=None, X=None, decimal=5, **kwargs):
        """
        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            X (tuple, list, np.ndarray): The features of datasets
            decimal (int): The number of fractional parts after the decimal point
            **kwargs ():
        """
        super().__init__(y_true, y_pred, decimal, **kwargs)
        if kwargs is None: kwargs = {}
        self.set_keyword_arguments(kwargs)
        self.X = X
        self.binary = True
        self.representor = "number"  # "number" or "string"
        self.le = None  # LabelEncoder

    def get_processed_data(self, y_true=None, y_pred=None, decimal=None):
        """
        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred)
            decimal (int, None): The number of fractional parts after the decimal point

        Returns:
            y_true_final: y_true used in evaluation process.
            y_pred_final: y_pred used in evaluation process
            one_dim: is y_true has 1 dimensions or not
            decimal: The number of fractional parts after the decimal point
        """
        decimal = self.decimal if decimal is None else decimal
        if (y_true is not None) and (y_pred is not None):
            y_true, y_pred, binary, representor = format_classification_data(y_true, y_pred)
        else:
            if (self.y_true is not None) and (self.y_pred is not None):
                y_true, y_pred, binary, representor = format_classification_data(self.y_true, self.y_pred)
            else:
                raise ValueError("y_true or y_pred is None. You need to pass y_true and y_pred to object creation or function called.")
        return y_true, y_pred, binary, representor, decimal

    def get_processed_y_pred(self, y_pred=None):
        if y_pred is not None:
            return format_clustering_label(y_pred)
        else:
            if self.y_pred is None:
                raise ValueError("To calculate clustering metrics, you need to pass y_pred")
            else:
                return format_clustering_label(self.y_pred)

    def check_X(self, X):
        if X is None:
            if self.X is None:
                raise ValueError("To calculate internal metrics, you need to pass X.")
            else:
                return self.X
        return X

    def ball_hall_index(self, X=None, y_pred=None, **kwargs):
        """
        The Ball-Hall Index (1995) is the mean of the mean dispersion across all clusters.
        The **largest difference** between successive clustering levels indicates the optimal number of clusters.

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.

        Returns:
            BH (float): The Ball-Hall index
        """
        X = self.check_X(X)
        y_pred = self.get_processed_y_pred(y_pred)
        n_classes = len(np.unique(y_pred))
        wgss = []
        ## For each cluster, find the centroid and then the within-group SSE
        for k in range(n_classes):
            centroid_mask = y_pred == k
            cluster_k = X[centroid_mask]
            centroid = np.mean(cluster_k, axis=0)
            wgss.append(np.sum((cluster_k - centroid) ** 2))
        return np.sum(wgss) / n_classes

    def calinski_harabasz_score(self, X=None, y_pred=None, **kwargs):
        """
        Compute the Calinski and Harabasz (1974) index. It is also known as the Variance Ratio Criterion.
        The score is defined as ratio between the within-cluster dispersion and the between-cluster dispersion.

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.

        Returns:
            score (float): The resulting Calinski-Harabasz score.

        References:
        .. [1] `T. Calinski and J. Harabasz, 1974. "A dendrite method for cluster
            analysis". Communications in Statistics <https://www.tandfonline.com/doi/abs/10.1080/03610927408827101>`_
        """
        X = self.check_X(X)
        y_pred = self.get_processed_y_pred(y_pred)
        n_samples, n_vars = X.shape
        n_classes = len(np.unique(y_pred))
        denom= (general_sums_of_squares(X, y_pred)["WGSS"] * (n_classes - 1))
        numer = (general_sums_of_squares(X, y_pred)["BGSS"] * (n_samples - n_classes))
        return numer / denom

    def xie_beni_index(self, X=None, y_pred=None, **kwargs):
        """
        Computes the Xie-Beni index.

        The Xie-Beni index is an index of fuzzy clustering, but it is also applicable to crisp clustering.
        The numerator is the mean of the squared distances of all of the points with respect to their
        barycenter of the cluster they belong to. The denominator is the minimal squared distances between
        the points in the clusters. The **minimum** value indicates the best number of clusters.

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.

        Returns:
            xb (float): The Xie-Beni index
        """
        X = self.check_X(X)
        y_pred = self.get_processed_y_pred(y_pred)
        # Get the centroids
        centroids = get_centroids(X, y_pred)
        euc_distance_to_centroids = get_min_dist(X, centroids)
        WGSS = np.sum(euc_distance_to_centroids**2)
        # Computing the minimum squared distance to the centroids:
        MinSqDist = np.min(pdist(centroids, metric='sqeuclidean'))
        # Computing the XB index:
        xb = (1 / X.shape[0]) * (WGSS / MinSqDist)
        return xb

    BHI = ball_hall_index
    CHS = calinski_harabasz_score
    XBI = xie_beni_index
