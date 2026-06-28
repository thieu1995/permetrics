#!/usr/bin/env python
# Created by "Matt Q." at 23:05, 27/10/2022 --------%
#       Github: https://github.com/N3uralN3twork    %
#                                                   %
# Improved by: "Thieu" at 11:45, 25/07/2023 --------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from permetrics.evaluator import Evaluator
from permetrics.utils import data_util as du
from permetrics.utils import cluster_util as cu


class ClusteringMetric(Evaluator):
    """
    Defines a ClusteringMetric class that hold all internal and external metrics for clustering problems

    + An extension of scikit-learn metrics section, with the addition of many more internal metrics.
    + https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation

    Parameters
    ----------
    y_true: tuple, list, np.ndarray, default = None
        The ground truth values. This is for calculating external metrics

    y_pred: tuple, list, np.ndarray, default = None
        The prediction values. This is for both calculating internal and external metrics

    X: tuple, list, np.ndarray, default = None
        The features of datasets. This is for calculating internal metrics

    force_finite: bool, default = True
        When result is not finite, it can be NaN or Inf. Their result will be replaced by `finite_value`

    finite_value: float, default = None
        The value that used to replace the infinite value or NaN value.
    """

    SUPPORT = {
        "BHI": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "CHI": {"type": "max", "range": "[0, +inf)", "best": "unknown"},
        "XBI": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "DBI": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "BRI": {"type": "min", "range": "(-inf, +inf)", "best": "unknown"},
        "DRI": {"type": "max", "range": "[1, +inf)", "best": "unknown"},
        "KDI": {"type": "max", "range": "(-inf, +inf)", "best": "unknown"},
        "DI": {"type": "max", "range": "[0, +inf)", "best": "unknown"},
        "LDRI": {"type": "max", "range": "(-inf, +inf)", "best": "unknown"},
        "LSRI": {"type": "max", "range": "(-inf, +inf)", "best": "unknown"},
        "SI": {"type": "max", "range": "[-1, 1]", "best": "1"},
        "SSEI": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "MSEI": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "DHI": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "BI": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "RSI": {"type": "max", "range": "[0, 1]", "best": "1"},
        "DBCVI": {"type": "max", "range": "[-1, 1]", "best": "1"},
        "HI": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "MIS": {"type": "max", "range": "[0, +inf)", "best": "unknown"},
        "NMIS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "RaS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "ARS": {"type": "max", "range": "[-1, 1]", "best": "1"},
        "FMS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "HS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "CS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "VMS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "PrS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "ReS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "FS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "CDS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "HGS": {"type": "max", "range": "[-1, 1]", "best": "1"},
        "JS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "KS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "MNS": {"type": "max", "range": "(-inf, +inf)", "best": "unknown"},
        "PhS": {"type": "max", "range": "[-1, 1]", "best": "1"},
        "RTS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "RRS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "SS1S": {"type": "max", "range": "[0, 1]", "best": "1"},
        "SS2S": {"type": "max", "range": "[0, 1]", "best": "1"},
        "PuS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "EnS": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "TauS": {"type": "max", "range": "[-1, 1]", "best": "1"},
        "GAS": {"type": "max", "range": "[-1, 1]", "best": "1"},
        "GPS": {"type": "min", "range": "[0, 1]", "best": "0"},
    }

    def __init__(self, y_true=None, y_pred=None, X=None, force_finite=True, finite_value=None, **kwargs):
        super().__init__(y_true, y_pred, **kwargs)
        if kwargs is None: kwargs = {}
        self.set_keyword_arguments(kwargs)
        self.X = X
        self.le = None
        self.force_finite = force_finite
        self.finite_value = finite_value

    @staticmethod
    def get_support(name=None, verbose=True):
        if name == "all":
            if verbose:
                for key, value in ClusteringMetric.SUPPORT.items():
                    print(f"Metric {key} : {value}")
            return ClusteringMetric.SUPPORT
        if name not in list(ClusteringMetric.SUPPORT.keys()):
            raise ValueError(f"ClusteringMetric doesn't support metric named: {name}")
        else:
            if verbose:
                print(f"Metric {name}: {ClusteringMetric.SUPPORT[name]}")
            return ClusteringMetric.SUPPORT[name]

    def get_processed_external_data(self, y_true=None, y_pred=None, force_finite=None, finite_value=None):
        """
        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            force_finite (bool): Force the result as finite number
            finite_value (float): The finite number

        Returns:
            y_true_final: y_true used in evaluation process.
            y_pred_final: y_pred used in evaluation process
            le: label encoder object
            force_finite: Force the result as finite number
            finite_value: The finite number
        """
        force_finite = self.force_finite if force_finite is None else force_finite
        finite_value = self.finite_value if finite_value is None else finite_value

        # Prioritize parameters passed to the function; if none are available, retrieve them from the instance.
        yt = y_true if y_true is not None else self.y_true
        yp = y_pred if y_pred is not None else self.y_pred
        if yt is None or yp is None:
            raise ValueError("You need to pass y_true and y_pred to calculate external clustering metrics.")
        yt_final, yp_final, self.le = du.format_external_clustering_data(yt, yp)
        return yt_final, yp_final, self.le, force_finite, finite_value

    def get_processed_internal_data(self, y_pred=None, force_finite=None, finite_value=None):
        """
        Args:
            y_pred (tuple, list, np.ndarray): The prediction values
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            y_pred_final: y_pred used in evaluation process
            le: label encoder object
            force_finite
            finite_value
        """
        force_finite = self.force_finite if force_finite is None else force_finite
        finite_value = self.finite_value if finite_value is None else finite_value
        yp = y_pred if y_pred is not None else self.y_pred
        if yp is None:
            raise ValueError("You need to pass y_pred to calculate external clustering metrics.")
        y_pred, self.le = du.format_internal_clustering_data(yp)
        return y_pred, self.le, force_finite, finite_value

    def check_X(self, X):
        data = X if X is not None else self.X
        if data is None:
            raise ValueError("You need to pass X to calculate internal clustering metrics.")
        features_arr = np.asarray(data)
        ## Check if the ndim is exactly 2
        if features_arr.ndim != 2:
            raise ValueError(f"Expected a 2D array, but got a {features_arr.ndim}D array instead.")
        ## Check if the array is empty (e.g., shape is (0, 0) or (5, 0))
        if features_arr.size == 0:
            raise ValueError("The provided 2D array is empty!")
        return features_arr

    def ball_hall_index(self, X=None, y_pred=None, **kwargs):
        """
        The Ball-Hall Index (1995) is the mean of the mean dispersion across all clusters.
        The **largest difference** between successive clustering levels indicates the optimal number of clusters.
        Smaller is better (Best = 0), Range=[0, +inf)

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.

        Returns:
            result (float): The Ball-Hall index
        """
        X = self.check_X(X)
        y_pred, _, _, _ = self.get_processed_internal_data(y_pred)
        return cu.calculate_ball_hall_index(X, y_pred)

    def calinski_harabasz_index(self, X=None, y_pred=None, force_finite=True, finite_value=0., **kwargs):
        """
        Compute the Calinski and Harabasz (1974) index. It is also known as the Variance Ratio Criterion.
        The score is defined as ratio between the within-cluster dispersion and the between-cluster dispersion.

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The resulting Calinski-Harabasz index.
        """
        X = self.check_X(X)
        y_pred, _, force_finite, finite_value = self.get_processed_internal_data(y_pred, force_finite, finite_value)
        return cu.calculate_calinski_harabasz_index(X, y_pred, force_finite, force_finite)

    def xie_beni_index(self, X=None, y_pred=None, force_finite=True, finite_value=1e10, **kwargs):
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
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Xie-Beni index
        """
        X = self.check_X(X)
        y_pred, _, force_finite, finite_value = self.get_processed_internal_data(y_pred, force_finite, finite_value)
        return cu.calculate_xie_beni_index(X, y_pred, force_finite, finite_value)

    def davies_bouldin_index(self, X=None, y_pred=None, force_finite=True, finite_value=1e10, **kwargs):
        """
        Computes the Davies-Bouldin index

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Davies-Bouldin index
        """
        X = self.check_X(X)
        y_pred, _, force_finite, finite_value = self.get_processed_internal_data(y_pred, force_finite, finite_value)
        return cu.calculate_davies_bouldin_index(X, y_pred, force_finite, finite_value)

    def banfeld_raftery_index(self, X=None, y_pred=None, force_finite=True, finite_value=1e10, **kwargs):
        """
        Computes the Banfeld-Raftery Index.

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Banfeld-Raftery Index
        """
        X = self.check_X(X)
        y_pred, _, force_finite, finite_value = self.get_processed_internal_data(y_pred, force_finite, finite_value)
        return cu.calculate_banfeld_raftery_index(X, y_pred, force_finite, finite_value)

    def det_ratio_index(self, X=None, y_pred=None, force_finite=True, finite_value=0., **kwargs):
        """
        Computes the Det-Ratio index

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Det-Ratio index
        """
        X = self.check_X(X)
        y_pred, _, force_finite, finite_value = self.get_processed_internal_data(y_pred, force_finite, finite_value)
        return cu.calculate_det_ratio_index(X, y_pred, force_finite, finite_value)

    def ksq_detw_index(self, X=None, y_pred=None, use_normalized=True, **kwargs):
        """
        Computes the Ksq-DetW Index

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.
            use_normalized (bool): We normalize the scatter matrix before calculate the Det to reduce the value, default=True

        Returns:
            result (float): The Ksq-DetW Index
        """
        X = self.check_X(X)
        y_pred, _, _, _ = self.get_processed_internal_data(y_pred)
        return cu.calculate_ksq_detw_index(X, y_pred, use_normalized)

    def log_det_ratio_index(self, X=None, y_pred=None, force_finite=True, finite_value=-1e10, **kwargs):
        """
        Computes the Log Det Ratio Index

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Log Det Ratio Index
        """
        X = self.check_X(X)
        y_pred, _, force_finite, finite_value = self.get_processed_internal_data(y_pred, force_finite, finite_value)
        return cu.calculate_log_det_ratio_index(X, y_pred, force_finite, finite_value)

    def dunn_index(self, X=None, y_pred=None, use_modified=True, force_finite=True, finite_value=0., **kwargs):
        """
        Computes the Dunn Index

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.
            use_modified (bool): The modified version we proposed to speed up the computational time for this metric, default=True
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Dunn Index
        """
        X = self.check_X(X)
        y_pred, _, force_finite, finite_value = self.get_processed_internal_data(y_pred, force_finite, finite_value)
        return cu.calculate_dunn_index(X, y_pred, use_modified, force_finite, finite_value)

    def log_ss_ratio_index(self, X=None, y_pred=None, force_finite=True, finite_value=-1e10, **kwargs):
        """
        Computes the Log SS Ratio Index

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Log SS Ratio Index
        """
        X = self.check_X(X)
        y_pred, _, force_finite, finite_value = self.get_processed_internal_data(y_pred, force_finite, finite_value)
        n_clusters = len(np.unique(y_pred))
        if n_clusters == 1:
            if self.force_finite:
                return self.finite_value
            else:
                raise ValueError("The Log SS Ratio Index is undefined when y_pred has only 1 cluster.")
        centers, _ = cu.compute_barycenters(X, y_pred)
        bgss = cu.compute_BGSS(X, y_pred)
        wgss = cu.compute_WGSS(X, y_pred)
        return np.log(bgss/wgss)

    def silhouette_index(self, X=None, y_pred=None, multi_output=False, force_finite=True, finite_value=-1., chunk_size=5000, **kwargs):
        """
        Computes the Silhouette Index

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.
            multi_output (bool): Returned scores for each cluster, default=False
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.
            chunk_size (int): Split original data to chunk_size to avoid OOM problem

        Returns:
            result (float): The Silhouette Index
        """
        X = self.check_X(X)
        y_pred, _, force_finite, finite_value = self.get_processed_internal_data(y_pred, force_finite, finite_value)
        return cu.calculate_silhouette_index(X, y_pred, chunk_size=chunk_size, multi_output=multi_output,
                                             force_finite=force_finite, finite_value=finite_value)

    def sum_squared_error_index(self, X=None, y_pred=None, **kwarg):
        """
        Computes the Sum of Squared Error Index

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.

        Returns:
            result (float): The Sum of Squared Error Index
        """
        X = self.check_X(X)
        y_pred, _, _, _ = self.get_processed_internal_data(y_pred)
        return cu.calculate_sum_squared_error_index(X, y_pred)

    def mean_squared_error_index(self, X=None, y_pred=None, **kwarg):
        """
        Computes the Mean Squared Error Index
        MSEI measures the mean of squared distances between each data point and its corresponding centroid or cluster center.

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.

        Returns:
            result (float): The Mean Squared Error Index
        """
        X = self.check_X(X)
        y_pred, _, _, _ = self.get_processed_internal_data(y_pred)
        return cu.calculate_mean_squared_error_index(X, y_pred)

    def duda_hart_index(self, X=None, y_pred=None, chunk_size=5000, force_finite=True, finite_value=1e10, **kwargs):
        """
        Computes the Duda Index or Duda-Hart index

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.
            chunk_size (int): Split original data to chunk_size to avoid OOM problem
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Duda-Hart index
        """
        X = self.check_X(X)
        y_pred, _, force_finite, finite_value = self.get_processed_internal_data(y_pred, force_finite, finite_value)
        return cu.calculate_duda_hart_index(X, y_pred, chunk_size, force_finite, finite_value)

    def beale_index(self, X=None, y_pred=None, force_finite=True, finite_value=1e10, **kwarg):
        """
        Computes the Beale Index

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Beale Index
        """
        X = self.check_X(X)
        y_pred, _, force_finite, finite_value = self.get_processed_internal_data(y_pred, force_finite, finite_value)
        return cu.calculate_beale_index(X, y_pred, force_finite, finite_value)

    def r_squared_index(self, X=None, y_pred=None, **kwarg):
        """
        Computes the R-squared index

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.

        Returns:
            result (float): The R-squared index
        """
        X = self.check_X(X)
        y_pred, _, _, _ = self.get_processed_internal_data(y_pred)
        return cu.calculate_r_squared_index(X, y_pred)

    def density_based_clustering_validation_index(self, X=None, y_pred=None, force_finite=True, finite_value=0., return_type="global", **kwarg):
        """
        Computes the Density-based Clustering Validation Index

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.
            return_type (str): The output type. Can be "global", "per-cluster", or "both". Default is "global".

        Returns:
            float or dict or tuple:
                - If "global": Returns the overall DBCV score (float).
                - If "per-cluster": Returns a dictionary mapping valid cluster labels to their individual validity scores.
                - If "both": Returns a tuple (global_score, per_cluster_dict).
        """
        X = self.check_X(X)
        y_pred, _, force_finite, finite_value = self.get_processed_internal_data(y_pred, force_finite, finite_value)
        gb, per_cluster_dict = cu.calculate_dbcv_score(X, y_pred, force_finite, finite_value)
        if return_type == "per-cluster":
            return per_cluster_dict
        elif return_type == "both":
            return gb, per_cluster_dict
        else:
            return gb

    def hartigan_index(self, X=None, y_pred=None, force_finite=True, finite_value=1e10, **kwarg):
        """
        Computes the Hartigan index for a clustering solution.

        Args:
            X (array-like of shape (n_samples, n_features)):
                A list of `n_features`-dimensional data points. Each row corresponds to a single data point.
            y_pred (array-like of shape (n_samples,)): Predicted labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Hartigan index
        """
        X = self.check_X(X)
        y_pred, _, force_finite, finite_value = self.get_processed_internal_data(y_pred, force_finite, finite_value)
        return cu.calculate_hartigan_index(X, y_pred, force_finite, finite_value)

    def mutual_info_score(self, y_true=None, y_pred=None, **kwargs):
        """
        Computes the Mutual Information score.

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.

        Returns:
            result (float): The Mutual Information score
        """
        y_true, y_pred, _, _, _ = self.get_processed_external_data(y_true, y_pred)
        return cu.calculate_mutual_info_score(y_true, y_pred)

    def normalized_mutual_info_score(self, y_true=None, y_pred=None, force_finite=True, finite_value=0., **kwargs):
        """
        Computes the normalized mutual information
        It is a variation of the mutual information score that normalizes the result to take values between 0 and 1.
        It is defined as the mutual information divided by the average entropy of the true and predicted clusterings.
        Bigger is better (Best = 1), Range = [0, 1]

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The normalized mutual information score.
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_normalized_mutual_info_score(y_true, y_pred, force_finite, finite_value)

    def rand_score(self, y_true=None, y_pred=None, **kwargs):
        """
        Computes the Rand score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.

        Returns:
            result (float): The rand score.
        """
        y_true, y_pred, _, _, _ = self.get_processed_external_data(y_true, y_pred)
        return cu.calculate_rand_score(y_true, y_pred)

    def adjusted_rand_score(self, y_true=None, y_pred=None, **kwargs):
        """
        Computes the Adjusted rand score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.

        Returns:
            result (float): The Adjusted rand score
        """
        y_true, y_pred, _, _, _ = self.get_processed_external_data(y_true, y_pred)
        return cu.calculate_adjusted_rand_score(y_true, y_pred)

    def fowlkes_mallows_score(self, y_true=None, y_pred=None, force_finite=True, finite_value=0., **kwargs):
        """
        Computes the Fowlkes-Mallows score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Fowlkes-Mallows score
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_fowlkes_mallows_score(y_true, y_pred, force_finite, finite_value)

    def homogeneity_score(self, y_true=None, y_pred=None, force_finite=True, finite_value=0.,**kwargs):
        """
        Computes the Homogeneity Score

        It measures the extent to which each cluster contains only data points that belong to a single class or category.
        In other words, homogeneity assesses whether all the data points in a cluster are members of the same true class or label.
        A higher homogeneity score indicates better clustering results, where each cluster corresponds well to a single ground truth class.

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Homogeneity Score
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_homogeneity_score(y_true, y_pred, force_finite, finite_value)

    def completeness_score(self, y_true=None, y_pred=None, force_finite=True, finite_value=0.,**kwargs):
        """
        Computes the Completeness Score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The completeness score.
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_completeness_score(y_true, y_pred, force_finite, finite_value)

    def v_measure_score(self, y_true=None, y_pred=None, beta=1.0, force_finite=True, finite_value=0., **kwargs):
        """
        Computes the V Measure Score

        It is a combination of two other metrics: homogeneity and completeness. Homogeneity measures whether all the
        data points in a given cluster belong to the same class. Completeness measures whether all the data points of a certain
        class are assigned to the same cluster. The V-measure combines these two metrics into a single score.

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            beta (float): The weight parameter, default = 1.0
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The V measure score
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_v_measure_score(y_true, y_pred, beta, force_finite, finite_value)

    def precision_score(self, y_true=None, y_pred=None, force_finite=True, finite_value=0., **kwargs):
        """
        Computes the Precision Score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Precision score
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_precision_score(y_true, y_pred, force_finite, finite_value)

    def recall_score(self, y_true=None, y_pred=None, force_finite=True, finite_value=0., **kwargs):
        """
        Computes the Recall Score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Recall score
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_recall_score(y_true, y_pred, force_finite, finite_value)

    def f_measure_score(self, y_true=None, y_pred=None, beta=1.0, force_finite=True, finite_value=0., **kwargs):
        """
        Computes the F-Measure score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            beta (float): The weight parameter, default = 1.0
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The F-Measure score
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_f_measure_score(y_true, y_pred, beta, force_finite, finite_value)

    def czekanowski_dice_score(self, y_true=None, y_pred=None, force_finite=True, finite_value=0., **kwargs):
        """
        Computes the Czekanowski-Dice score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Czekanowski-Dice score
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_czekanowski_dice_score(y_true, y_pred, force_finite, finite_value)

    def hubert_gamma_score(self, y_true=None, y_pred=None, force_finite=True, finite_value=0.0, **kwargs):
        """
        Computes the Hubert Gamma score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Hubert Gamma score
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_hubert_gamma_score(y_true, y_pred, force_finite, finite_value)

    def jaccard_score(self, y_true=None, y_pred=None, force_finite=True, finite_value=0.0, **kwargs):
        """
        Computes the Jaccard score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Jaccard score
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_jaccard_score(y_true, y_pred, force_finite, finite_value)

    def kulczynski_score(self, y_true=None, y_pred=None, force_finite=True, finite_value=0.0, **kwargs):
        """
        Computes the Kulczynski Score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Kulczynski score
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_kulczynski_score(y_true, y_pred, force_finite, finite_value)

    def mc_nemar_score(self, y_true=None, y_pred=None, force_finite=True, finite_value=0.0, **kwargs):
        """
        Computes the Mc Nemar score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Mc Nemar score
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_mc_nemar_score(y_true, y_pred, force_finite, finite_value)

    def phi_score(self, y_true=None, y_pred=None, force_finite=True, finite_value=0.0, **kwargs):
        """
        Computes the Phi score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Phi score
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_phi_score(y_true, y_pred, force_finite, finite_value)

    def rogers_tanimoto_score(self, y_true=None, y_pred=None, force_finite=True, finite_value=0.0, **kwargs):
        """
        Computes the Rogers-Tanimoto score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Rogers-Tanimoto score
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_rogers_tanimoto_score(y_true, y_pred, force_finite, finite_value)

    def russel_rao_score(self, y_true=None, y_pred=None, force_finite=True, finite_value=0.0, **kwargs):
        """
        Computes the Russel-Rao score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Russel-Rao score
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_russel_rao_score(y_true, y_pred, force_finite, finite_value)

    def sokal_sneath1_score(self, y_true=None, y_pred=None, force_finite=True, finite_value=0.0, **kwargs):
        """
        Computes the Sokal-Sneath 1 score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Sokal-Sneath 1 score
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_sokal_sneath1_score(y_true, y_pred, force_finite, finite_value)

    def sokal_sneath2_score(self, y_true=None, y_pred=None, force_finite=True, finite_value=0.0, **kwargs):
        """
        Computes the Sokal-Sneath 2 score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Sokal-Sneath 2 score
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_sokal_sneath2_score(y_true, y_pred, force_finite, finite_value)

    def purity_score(self, y_true=None, y_pred=None, **kwargs):
        """
        Computes the Purity score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.

        Returns:
            result (float): The Purity score
        """
        y_true, y_pred, _, _, _ = self.get_processed_external_data(y_true, y_pred)
        return cu.calculate_purity_score(y_true, y_pred)

    def entropy_score(self, y_true=None, y_pred=None, **kwargs):
        """
        Computes the Entropy score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.

        Returns:
            result (float): The Entropy score
        """
        y_true, y_pred, _, _, _ = self.get_processed_external_data(y_true, y_pred)
        return cu.calculate_entropy_score(y_true, y_pred)

    def tau_score(self, y_true=None, y_pred=None, force_finite=True, finite_value=0.0, **kwargs):
        """
        Computes the Tau Score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Tau Score
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_tau_score(y_true, y_pred, force_finite, finite_value)

    def gamma_score(self, y_true=None, y_pred=None, force_finite=True, finite_value=0.0, **kwargs):
        """
        Computes the Gamma Score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Gamma Score
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_gamma_score(y_true, y_pred, force_finite, finite_value)

    def gplus_score(self, y_true=None, y_pred=None, force_finite=True, finite_value=0.0, **kwargs):
        """
        Computes the Gplus Score

        Args:
            y_true (array-like): The true labels for each sample.
            y_pred (array-like): The predicted cluster labels for each sample.
            force_finite (bool): Make result as finite number
            finite_value (float): The value that used to replace the infinite value or NaN value.

        Returns:
            result (float): The Gplus Score
        """
        y_true, y_pred, _, force_finite, finite_value = self.get_processed_external_data(y_true, y_pred, force_finite, finite_value)
        return cu.calculate_gplus_score(y_true, y_pred, force_finite, finite_value)


    BHI = ball_hall_index
    CHI = calinski_harabasz_index
    XBI = xie_beni_index
    DBI = davies_bouldin_index
    BRI = banfeld_raftery_index
    KDI = ksq_detw_index
    DRI = det_ratio_index
    DI = dunn_index
    LDRI = log_det_ratio_index
    LSRI = log_ss_ratio_index
    SI = silhouette_index
    SSEI = sum_squared_error_index
    MSEI = mean_squared_error_index
    DHI = duda_hart_index
    BI = beale_index
    RSI = r_squared_index
    DBCVI = density_based_clustering_validation_index
    HI = hartigan_index
    MIS = mutual_info_score
    NMIS = normalized_mutual_info_score
    RaS = rand_score
    ARS = adjusted_rand_score
    FMS = fowlkes_mallows_score
    HS = homogeneity_score
    CS = completeness_score
    VMS = v_measure_score
    PrS = precision_score
    ReS = recall_score
    FS = f_measure_score
    CDS = czekanowski_dice_score
    HGS = hubert_gamma_score
    JS = jaccard_score
    KS = kulczynski_score
    MNS = mc_nemar_score
    PhS = phi_score
    RTS = rogers_tanimoto_score
    RRS = russel_rao_score
    SS1S = sokal_sneath1_score
    SS2S = sokal_sneath2_score
    PuS = purity_score
    EnS = entropy_score
    TauS = tau_score
    GAS = gamma_score
    GPS = gplus_score
