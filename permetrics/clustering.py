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
            result (float): The Ball-Hall index
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

    def calinski_harabasz_index(self, X=None, y_pred=None, **kwargs):
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
            result (float): The resulting Calinski-Harabasz index.

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
            result (float): The Xie-Beni index
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

    def banfeld_raftery_index(self, X=None, y_pred=None, **kwargs):
        """
        Computes the Banfeld-Raftery Index.
        This index is the weighted sum of the logarithms of the traces of the variancecovariance matrix of each cluster

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.

        Returns:
            result (float): The Banfeld-Raftery Index
        """
        X = self.check_X(X)
        y_pred, _, decimal = self.get_processed_internal_data(y_pred)
        clusters_dict, cluster_sizes_dict = cu.compute_clusters(y_pred)
        cc = 0.0
        for k in clusters_dict.keys():
            X_k = X[clusters_dict[k]]
            cluster_dispersion = np.trace(cu.compute_WG(X_k)) / cluster_sizes_dict[k]
            if cluster_sizes_dict[k] > 1:
                cc += cluster_sizes_dict[k] * np.log(cluster_dispersion)
        return np.round(cc, decimal)

    def davies_bouldin_index(self, X=None, y_pred=None, **kwargs):
        """
        Computes the Davies-Bouldin index

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.

        Returns:
            result (float): The Davies-Bouldin index
        """
        X = self.check_X(X)
        y_pred, _, decimal = self.get_processed_internal_data(y_pred)
        clusters_dict, cluster_sizes_dict = cu.compute_clusters(y_pred)
        centers, _ = cu.compute_barycenters(X, y_pred)
        n_clusters = len(clusters_dict)
        # Calculate delta for each cluster
        delta = {}
        for k in range(n_clusters):
            X_k = X[clusters_dict[k]]
            delta[k] = np.mean(np.linalg.norm(X_k - centers[k], axis=1))
        # Calculate the Davies-Bouldin index
        cc = 0.0
        for kdx in range(n_clusters):
            list_dist = []
            for jdx in range(n_clusters):
                if jdx != kdx:
                    m = (delta[kdx] + delta[jdx]) / np.linalg.norm(centers[kdx] - centers[jdx])
                    list_dist.append(m)
            cc += np.max(list_dist)
        return np.round(cc/n_clusters, decimal)

    def det_ratio_index(self, X=None, y_pred=None, **kwargs):
        """
        Computes the Det-Ratio index

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.

        Returns:
            result (float): The Det-Ratio index
        """
        X = self.check_X(X)
        y_pred, _, decimal = self.get_processed_internal_data(y_pred)
        clusters_dict, cluster_sizes_dict = cu.compute_clusters(y_pred)
        centers, _ = cu.compute_barycenters(X, y_pred)
        T = cu.compute_WG(X)
        scatter_matrices = np.zeros((X.shape[1], X.shape[1]))     # shape of (n_features, n_features)
        for label, indices in clusters_dict.items():
            # Retrieve data points for the current cluster
            X_k = X[indices]
            # Compute within-group scatter matrix for the current cluster
            scatter_matrices += cu.compute_WG(X_k)
        cc = np.linalg.det(T) / np.linalg.det(scatter_matrices)
        return np.round(cc, decimal)

    def dunn_index(self, X=None, y_pred=None, **kwargs):
        """
        Computes the Dunn Index

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.

        Returns:
            result (float): The Dunn Index
        """
        X = self.check_X(X)
        y_pred, _, decimal = self.get_processed_internal_data(y_pred)
        clusters_dict, cluster_sizes_dict = cu.compute_clusters(y_pred)
        n_clusters = len(clusters_dict)
        # Calculate dmin
        dmin = np.inf
        for kdx in range(n_clusters-1):
            for k0 in range(kdx + 1, n_clusters):
                t1 = X[clusters_dict[kdx]]
                t2 = X[clusters_dict[k0]]
                dkk = cu.cdist(t1, t2, metric='euclidean')
                dmin = min(dmin, np.min(dkk))
        # Calculate dmax
        dmax = 0.0
        for kdx in range(n_clusters):
            max_d_cluster = 0.0
            for idx in range(len(clusters_dict[kdx])-1):
                for jdx in range(idx + 1, len(clusters_dict[kdx])):
                    dk = np.linalg.norm(X[clusters_dict[kdx]][idx] - X[clusters_dict[kdx]][jdx])
                    max_d_cluster = max(max_d_cluster, dk)
            dmax = max(dmax, max_d_cluster)
        return np.round(dmin/dmax, decimal)

    def ksq_detw_index(self, X=None, y_pred=None, **kwargs):
        """
        Computes the Ksq-DetW Index

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.

        Returns:
            result (float): The Ksq-DetW Index
        """
        X = self.check_X(X)
        y_pred, _, decimal = self.get_processed_internal_data(y_pred)
        clusters_dict, cluster_sizes_dict = cu.compute_clusters(y_pred)
        centers, _ = cu.compute_barycenters(X, y_pred)
        scatter_matrices = np.zeros((X.shape[1], X.shape[1]))     # shape of (n_features, n_features)
        for label, indices in clusters_dict.items():
            X_k = X[indices]
            scatter_matrices += cu.compute_WG(X_k)
        cc = len(clusters_dict)**2 * np.linalg.det(scatter_matrices)
        return np.round(cc, decimal)

    def log_det_ratio_index(self, X=None, y_pred=None, **kwargs):
        """
        Computes the Log Det Ratio Index

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.

        Returns:
            result (float): The Log Det Ratio Index
        """
        X = self.check_X(X)
        y_pred, _, decimal = self.get_processed_internal_data(y_pred)
        clusters_dict, cluster_sizes_dict = cu.compute_clusters(y_pred)
        centers, _ = cu.compute_barycenters(X, y_pred)
        T = cu.compute_WG(X)
        WG = np.zeros((X.shape[1], X.shape[1]))     # shape of (n_features, n_features)
        for label, indices in clusters_dict.items():
            X_k = X[indices]
            WG += cu.compute_WG(X_k)
        cc = X.shape[0] * np.log(np.linalg.det(T) / np.linalg.det(WG))
        return np.round(cc, decimal)

    def log_ss_ratio_index(self, X=None, y_pred=None, **kwargs):
        """
        Computes the Log SS Ratio Index

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.

        Returns:
            result (float): The Log SS Ratio Index
        """
        X = self.check_X(X)
        y_pred, _, decimal = self.get_processed_internal_data(y_pred)
        centers, _ = cu.compute_barycenters(X, y_pred)
        bgss = cu.compute_BGSS(X, y_pred)
        wgss = cu.compute_WGSS(X, y_pred)
        return np.round(np.log(bgss/wgss), decimal)

    def silhouette_index(self, X=None, y_pred=None, **kwarg):
        """
        Computes the Silhouette Index

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.

        Returns:
            result (float): The Silhouette Index
        """
        X = self.check_X(X)
        y_pred, _, decimal = self.get_processed_internal_data(y_pred)
        dm = cu.distance_matrix(X, X)
        silhouette_scores = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            a = np.mean(dm[i, y_pred == y_pred[i]])  # Cohesion
            b_values = [np.mean(dm[i, y_pred == label]) for label in np.unique(y_pred) if label != y_pred[i]]
            b = np.min(b_values) if len(b_values) > 0 else 0  # Separation
            silhouette_scores[i] = (b - a) / max(a, b)
        return np.mean(silhouette_scores)

    def baker_hubert_gamma_index(self, X=None, y_pred=None, **kwargs):
        """
        Computes the Baker-Hubert Gamma index
        TODO: Calculate based on O(N^2) of samples --> Very slow

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.

        Returns:
            result (float): The Baker-Hubert Gamma index
        """
        X = self.check_X(X)
        y_pred, _, decimal = self.get_processed_internal_data(y_pred)
        num_samples, num_features = X.shape
        n_pairs = (num_samples * (num_samples - 1)) // 2
        distances = np.zeros(n_pairs)
        binary_vector = np.zeros(n_pairs)
        index = 0
        for idx in range(num_samples-1):
            for jdx in range(idx + 1, num_samples):
                distances[index] = np.linalg.norm(X[idx] - X[jdx])
                binary_vector[index] = 0 if y_pred[idx] == y_pred[jdx] else 1
                index += 1
        s_plus = 0
        s_minus = 0
        for idx in range(0, n_pairs-1):
            for jdx in range(idx+1, n_pairs):
                if binary_vector[idx] == 0 and binary_vector[jdx] == 1:
                    # For each within-cluster distance (B = 0), compare with between-cluster distances (B = 1).
                    s_plus += np.sum(distances[idx] < distances[jdx])
                    s_minus += np.sum(distances[idx] > distances[jdx])
        # Calculate the Gamma index
        denominator = s_plus + s_minus
        gamma_index = (s_plus - s_minus) / denominator if denominator != 0 else 0.0
        return np.round(gamma_index, decimal)

    def g_plus_index(self, X=None, y_pred=None, **kwargs):
        """
        Computes the G plus index
        TODO: Calculate based on O(N^2) of samples --> Very slow

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.

        Returns:
            result (float): The G plus index
        """
        X = self.check_X(X)
        y_pred, _, decimal = self.get_processed_internal_data(y_pred)
        num_samples, num_features = X.shape
        n_pairs = (num_samples * (num_samples - 1)) // 2
        distances = np.zeros(n_pairs)
        binary_vector = np.zeros(n_pairs)
        num_discordant_pairs = 0
        for idx in range(0, n_pairs-1):
            for jdx in range(idx+1, n_pairs):
                if binary_vector[idx] == 0 and binary_vector[jdx] == 1:
                    # For each within-cluster distance (B = 0), check if it is greater than between-cluster distances (B = 1).
                    num_discordant_pairs += int(distances[idx] > distances[jdx])
        # Calculate the G plus index
        g_p = 2 * num_discordant_pairs / (n_pairs * (n_pairs - 1))
        return np.round(g_p, decimal)


    BHI = ball_hall_index
    CHI = calinski_harabasz_index
    XBI = xie_beni_index
    BRI = banfeld_raftery_index
    DBI = davies_bouldin_index
    DRI = det_ratio_index
    DI = dunn_index
    KDI = ksq_detw_index
    LDRI = log_det_ratio_index
    LSRI = log_ss_ratio_index
    SI = silhouette_index

    BHGI = baker_hubert_gamma_index
    GPI = g_plus_index
