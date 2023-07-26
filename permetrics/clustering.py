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
from permetrics.utils import cluster_util as cu
# from permetrics.utils.cluster_util import (get_min_dist, get_centroids, general_sums_of_squares, pdist,
#                                 cdist, squareform, get_labels,
#                                 average_scattering, cluster_sep, density_between, density_clusters, pmatch)
from permetrics.utils.encoder import LabelEncoder


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
        self.le = None

    def get_processed_external_data(self, y_true=None, y_pred=None, decimal=None):
        """
        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            decimal (int, None): The number of fractional parts after the decimal point

        Returns:
            y_true_final: y_true used in evaluation process.
            y_pred_final: y_pred used in evaluation process
            le: label encoder object
            decimal: The number of fractional parts after the decimal point
        """
        decimal = self.decimal if decimal is None else decimal
        if y_pred is None:              # Check for function called
            if self.y_pred is None:     # Check for object of class called
                raise ValueError("You need to pass y_true and y_pred to calculate external clustering metrics.")
            else:
                if self.y_true is None:
                    # y_true, y_pred, self.le = format_internal_clustering_data(self.y_pred)
                    raise ValueError("You need to pass y_true and y_pred to calculate external clustering metrics.")
                else:
                    y_true, y_pred, self.le = format_external_clustering_data(self.y_true, self.y_pred)
        else:   # This is for function called, it will override object of class called
            if y_true is None:
                # y_true, y_pred, self.le = format_internal_clustering_data(y_pred)
                raise ValueError("You need to pass y_true and y_pred to calculate external clustering metrics.")
            else:
                y_true, y_pred, self.le = format_external_clustering_data(y_true, y_pred)
        return y_true, y_pred, self.le, decimal

    def get_processed_internal_data(self, y_pred=None, decimal=None):
        """
        Args:
            y_pred (tuple, list, np.ndarray): The prediction values
            decimal (int, None): The number of fractional parts after the decimal point

        Returns:
            y_pred_final: y_pred used in evaluation process
            le: label encoder object
            decimal: The number of fractional parts after the decimal point
        """
        decimal = self.decimal if decimal is None else decimal
        if y_pred is None:              # Check for function called
            if self.y_pred is None:     # Check for object of class called
                raise ValueError("You need to pass y_pred to calculate external clustering metrics.")
            else:
                y_pred, self.le = format_internal_clustering_data(self.y_pred)
        else:   # This is for function called, it will override object of class called
            y_pred, self.le = format_internal_clustering_data(y_pred)
        return y_pred, self.le, decimal

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
        y_pred, _, decimal = self.get_processed_internal_data(y_pred)
        n_clusters = len(np.unique(y_pred))
        wgss = 0
        ## For each cluster, find the centroid and then the within-group SSE
        for k in range(n_clusters):
            centroid_mask = y_pred == k
            cluster_k = X[centroid_mask]
            centroid = np.mean(cluster_k, axis=0)
            wgss += np.sum((cluster_k - centroid) ** 2)
        return np.round(wgss / n_clusters, decimal)

    def calinski_harabasz_score(self, X=None, y_pred=None, **kwargs):
        """
        Compute the Calinski and Harabasz (1974) index. It is also known as the Variance Ratio Criterion.
        The score is defined as ratio between the within-cluster dispersion and the between-cluster dispersion.

        Notes:
        ~~~~~~
            + This metric in scikit-learn library is wrong in calculate the intra_disp variable (WGSS)
            + https://github.com/scikit-learn/scikit-learn/blob/7f9bad99d/sklearn/metrics/cluster/_unsupervised.py#L351C1-L351C1

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
        y_pred, _, decimal = self.get_processed_internal_data(y_pred)
        n_samples, n_vars = X.shape
        n_clusters = len(np.unique(y_pred))
        numer = cu.compute_BGSS(X, y_pred) * (n_samples - n_clusters)
        denom = cu.compute_WGSS(X, y_pred) * (n_clusters - 1)
        return np.round(numer / denom, decimal)

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
        y_pred, _, decimal = self.get_processed_internal_data(y_pred)
        # Get the centroids
        centroids = cu.get_centroids(X, y_pred)
        euc_distance_to_centroids = cu.get_min_dist(X, centroids)
        WGSS = np.sum(euc_distance_to_centroids**2)
        # Computing the minimum squared distance to the centroids:
        MinSqDist = np.min(cu.pdist(centroids, metric='sqeuclidean'))
        # Computing the XB index:
        xb = (1 / X.shape[0]) * (WGSS / MinSqDist)
        return xb
