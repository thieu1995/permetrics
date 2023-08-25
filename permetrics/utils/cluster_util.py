#!/usr/bin/env python
# Created by "Matt Q." at 23:05, 27/10/2022 --------%
#       Github: https://github.com/N3uralN3twork    %
#                                                   %
# Improved by: "Thieu" at 17:10, 25/07/2023 --------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial import distance_matrix
from scipy.stats import entropy as calculate_entropy
from scipy.sparse import coo_matrix
from collections import Counter


def compute_clusters(labels):
    """
    Get the dict of clusters and dict of cluster size
    """
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    dict_clusters = {label: np.where(labels == label)[0] for label in unique_labels}
    dict_cluster_sizes = {label: count for label, count in zip(unique_labels, label_counts)}
    return dict_clusters, dict_cluster_sizes


def compute_barycenters(X, labels):
    """
    Get the barycenter for each cluster and barycenter for all observations

    Args:
        X (np.ndarray): The features of datasets
        labels (np.ndarray): The predicted labels

    Returns:
        barycenters (np.ndarray): The barycenter for each clusters in form of matrix
        overall_barycenter (np.ndarray): the barycenter for all observations
    """
    n_samples, n_features = X.shape
    list_clusters = np.unique(labels)
    ## Mask mapping each class to its members.
    centroids = np.empty((len(list_clusters), n_features), dtype=np.float64)
    for idx, k in enumerate(list_clusters):
        centroids[idx] = X[labels == k].mean(axis=0)
    return centroids, np.mean(X, axis=0)


def compute_WG(X):
    # Compute the scatter matrix WG using Eq.11
    # Centering the column vectors
    means = np.mean(X, axis=0)
    centered_X = X - means
    # Computing the scatter matrix
    scatter_matrix = np.dot(centered_X.T, centered_X)
    return scatter_matrix


def compute_TSS(X):
    # The total scattering TSS (total sum of squares)
    # Computing the scatter matrix
    scatter_matrix = compute_WG(X)
    # Computing the total sum of squares (TSS)
    TSS = np.trace(scatter_matrix)
    return TSS


def compute_WGSS(X, labels):
    """
    Calculate the pooled within-cluster sum of squares WGSS or The within-cluster variance
    """
    barycenters, overall_barycenter = compute_barycenters(X, labels)
    n_clusters = len(barycenters)
    within_var = 0.0
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_mean = np.mean(X[cluster_mask], axis=0)
        within_var += np.sum(np.sum((X[cluster_mask] - cluster_mean) ** 2, axis=1))
    return within_var


def compute_BGSS(X, labels):
    """
    The between-group dispersion BGSS or between-cluster variance
    """
    barycenters, overall_barycenter = compute_barycenters(X, labels)
    n_clusters = len(barycenters)
    # Calculate the overall mean of the data
    overall_mean = np.mean(X, axis=0)

    # # Calculate between-cluster variance and cluster sizes
    # cluster_sizes = np.bincount(labels, minlength=n_clusters)
    # cluster_means = np.array([np.mean(X[labels == i], axis=0) for i in range(n_clusters)])
    # between_var = np.sum(cluster_sizes * np.sum((cluster_means - overall_mean) ** 2, axis=1))
    # return between_var

    # Calculate the between-cluster variance
    between_var = 0.0
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_size = np.sum(cluster_mask)
        cluster_mean = np.mean(X[cluster_mask], axis=0)
        between_var += cluster_size * np.sum((cluster_mean - overall_mean) ** 2)
    return between_var


def compute_contingency_matrix(y_true, y_pred):
    unique_true, true_inverse = np.unique(y_true, return_inverse=True)
    unique_pred, pred_inverse = np.unique(y_pred, return_inverse=True)
    contingency = coo_matrix((np.ones_like(y_true), (true_inverse, pred_inverse)),
                             shape=(len(unique_true), len(unique_pred)), dtype=np.int64).toarray()
    return contingency


def compute_confusion_matrix(y_true, y_pred, normalize=False):
    """
    Computes the confusion matrix for a clustering problem given the true labels and the predicted labels.
    http://cran.nexr.com/web/packages/clusterCrit/vignettes/clusterCrit.pdf
    """
    n = len(y_true)
    yy = yn = ny = nn = 0.
    for i in range(n - 1):
        y_true_diff = y_true[i + 1:] == y_true[i]
        y_pred_diff = y_pred[i + 1:] == y_pred[i]
        yy += np.sum(y_true_diff & y_pred_diff)
        yn += np.sum(y_true_diff & ~y_pred_diff)
        ny += np.sum(~y_true_diff & y_pred_diff)
        nn += np.sum(~y_true_diff & ~y_pred_diff)
    res = np.array([yy, yn, ny, nn], dtype=np.int64)
    if normalize:
        return res / np.sum(res)
    return res


def calculate_sum_squared_error_index(X=None, y_pred=None, decimal=6):
    centers, _ = compute_barycenters(X, y_pred)
    centroid_distances = centers[y_pred]
    squared_distances = np.sum((X - centroid_distances) ** 2, axis=1)
    return np.round(np.sum(squared_distances), decimal)


def calculate_mean_squared_error_index(X=None, y_pred=None, decimal=6):
    centers, _ = compute_barycenters(X, y_pred)
    centroid_distances = centers[y_pred]
    squared_distances = np.sum((X - centroid_distances) ** 2, axis=1)
    return np.round(np.mean(squared_distances), decimal)


def calculate_ball_hall_index(X=None, y_pred=None, decimal=6):
    n_clusters = len(np.unique(y_pred))
    wgss = 0
    ## For each cluster, find the centroid and then the within-group SSE
    for k in range(n_clusters):
        centroid_mask = y_pred == k
        cluster_k = X[centroid_mask]
        centroid = np.mean(cluster_k, axis=0)
        wgss += np.sum((cluster_k - centroid) ** 2)
    return np.round(wgss / n_clusters, decimal)


def calculate_calinski_harabasz_index(X=None, y_pred=None, decimal=6, raise_error=True, raise_value=0.0):
    n_samples, _ = X.shape
    n_clusters = len(np.unique(y_pred))
    if n_clusters == 1:
        if raise_error:
            raise ValueError("The Calinski-Harabasz index is undefined when y_pred has only 1 cluster.")
        else:
            return raise_value
    overall_mean = np.mean(X, axis=0)
    # Calculate between-cluster variance and cluster sizes
    cluster_sizes = np.bincount(y_pred, minlength=n_clusters)
    cluster_means = np.array([np.mean(X[y_pred == i], axis=0) for i in range(n_clusters)])
    between_var = np.sum(cluster_sizes * np.sum((cluster_means - overall_mean) ** 2, axis=1))
    # Calculate within-cluster variance
    within_var = np.sum((X - cluster_means[y_pred]) ** 2)
    # Calculate the CH Index
    res = (between_var / within_var) * ((n_samples - n_clusters) / (n_clusters - 1))
    return np.round(res, decimal)


def calculate_xie_beni_index(X=None, y_pred=None, decimal=6, raise_error=True, raise_value=np.inf):
    n_clusters = len(np.unique(y_pred))
    if n_clusters == 1:
        if raise_error:
            raise ValueError("The Xie-Beni index is undefined when y_pred has only 1 cluster.")
        else:
            return raise_value
    # Get the centroids
    centroids, _ = compute_barycenters(X, y_pred)
    wgss = np.sum(np.min(cdist(X, centroids, metric='euclidean'), axis=1) ** 2)
    # Computing the minimum squared distance to the centroids:
    MinSqDist = np.min(pdist(centroids, metric='sqeuclidean'))
    res = (wgss / X.shape[0]) / MinSqDist
    return np.round(res, decimal)


def calculate_banfeld_raftery_index(X=None, y_pred=None, decimal=6, raise_error=True, raise_value=np.inf):
    clusters_dict, cluster_sizes_dict = compute_clusters(y_pred)
    cc = 0.0
    for k in clusters_dict.keys():
        X_k = X[clusters_dict[k]]
        cluster_dispersion = np.trace(compute_WG(X_k)) / cluster_sizes_dict[k]
        if cluster_sizes_dict[k] == 1:
            if raise_error:
                raise ValueError("The Banfeld-Raftery index is undefined when at least 1 cluster has only 1 sample.")
            else:
                return raise_value
        if cluster_sizes_dict[k] > 1:
            cc += cluster_sizes_dict[k] * np.log(cluster_dispersion)
    return np.round(cc, decimal)


def calculate_davies_bouldin_index(X=None, y_pred=None, decimal=6, raise_error=True, raise_value=np.inf):
    clusters_dict, cluster_sizes_dict = compute_clusters(y_pred)
    centers, _ = compute_barycenters(X, y_pred)
    n_clusters = len(clusters_dict)
    if n_clusters == 1:
        if raise_error:
            raise ValueError("The Davies-Bouldin index is undefined when y_pred has only 1 cluster.")
        else:
            return raise_value
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
    return np.round(cc / n_clusters, decimal)


def calculate_det_ratio_index(X=None, y_pred=None, decimal=6, raise_error=True, raise_value=0.0):
    clusters_dict, cluster_sizes_dict = compute_clusters(y_pred)
    centers, _ = compute_barycenters(X, y_pred)
    T = compute_WG(X)
    scatter_matrices = np.zeros((X.shape[1], X.shape[1]))  # shape of (n_features, n_features)
    for label, indices in clusters_dict.items():
        # Retrieve data points for the current cluster
        X_k = X[indices]
        # Compute within-group scatter matrix for the current cluster
        scatter_matrices += compute_WG(X_k)
    t1 = np.linalg.det(scatter_matrices)
    if t1 == 0:
        if raise_error:
            raise ValueError("The Det-Ratio index is undefined when determinant of matrix is 0.")
        else:
            return raise_value
    return np.round(np.linalg.det(T) / t1, decimal)


def calculate_dunn_index(X=None, y_pred=None, decimal=6, use_modified=True, raise_error=True, raise_value=0.0):
    centers, _ = compute_barycenters(X, y_pred)
    n_clusters = len(centers)
    if n_clusters == 1:
        if raise_error:
            raise ValueError("The Dunn index is undefined when y_pred has only 1 cluster.")
        else:
            return raise_value
    # Calculate dmin
    dmin = np.inf
    if use_modified:
        for k0 in range(n_clusters - 1):
            for k1 in range(k0 + 1, n_clusters):
                points = X[y_pred == k1]
                dkk = np.min(cdist(points, centers[k0].reshape(1, -1), metric='euclidean'))
                dmin = min(dmin, np.min(dkk))
    else:
        for kdx in range(n_clusters - 1):
            for k0 in range(kdx + 1, n_clusters):
                points1 = X[y_pred == kdx]
                points2 = X[y_pred == k0]
                dkk = cdist(points1, points2, metric='euclidean')
                dmin = min(dmin, np.min(dkk))
    # Calculate dmax
    dmax = 0.0
    for kdx in range(n_clusters):
        points = X[y_pred == kdx]
        dk = np.max(pdist(points, metric="euclidean"))
        dmax = max(dmax, dk)
    return np.round(dmin / dmax, decimal)


def calculate_ksq_detw_index(X=None, y_pred=None, decimal=6, use_normalized=True):
    centers, _ = compute_barycenters(X, y_pred)
    scatter_matrices = np.zeros((X.shape[1], X.shape[1]))  # shape of (n_features, n_features)
    for kdx in range(len(centers)):
        X_k = X[y_pred == kdx]
        scatter_matrices += compute_WG(X_k)
    if use_normalized:
        scatter_matrices = (scatter_matrices - np.min(scatter_matrices)) / (np.max(scatter_matrices) - np.min(scatter_matrices))
    res = len(centers) ** 2 * np.linalg.det(scatter_matrices)
    return np.round(res, decimal)


def calculate_log_det_ratio_index(X=None, y_pred=None, decimal=6, raise_error=True, raise_value=-np.inf):
    clusters_dict, cluster_sizes_dict = compute_clusters(y_pred)
    centers, _ = compute_barycenters(X, y_pred)
    T = compute_WG(X)
    WG = np.zeros((X.shape[1], X.shape[1]))  # shape of (n_features, n_features)
    for label, indices in clusters_dict.items():
        X_k = X[indices]
        WG += compute_WG(X_k)
    t2 = np.linalg.det(WG)
    if t2 == 0:
        if raise_error:
            raise ValueError("The Log Det Ratio Index is undefined when determinant of matrix WG is 0.")
        else:
            return raise_value
    t1 = np.linalg.det(T) / t2
    if t1 <= 0:
        if raise_error:
            raise ValueError("The Log Det Ratio Index is undefined when det(T)/det(WG) <= 0.")
        else:
            return raise_value
    return np.round(X.shape[0] * np.log(t1), decimal)


def calculate_silhouette_index_slow(X=None, y_pred=None, decimal=6):
    dm = distance_matrix(X, X)
    res = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        a = np.mean(dm[i, y_pred == y_pred[i]])  # Cohesion
        b_values = [np.mean(dm[i, y_pred == label]) for label in np.unique(y_pred) if label != y_pred[i]]
        b = np.min(b_values) if len(b_values) > 0 else 0  # Separation
        res[i] = (b - a) / max(a, b)
    return np.round(np.mean(res), decimal)


def calculate_silhouette_index(X=None, y_pred=None, decimal=6, multi_output=False, raise_error=True, raise_value=-1.0):
    unique_clusters = np.unique(y_pred)
    if len(unique_clusters) == 1:
        if raise_error:
            raise ValueError("The Silhouette Index is undefined when y_pred has only 1 cluster.")
        else:
            return raise_value
    num_clusters = len(unique_clusters)
    num_points = len(X)
    # Precompute pairwise distances
    pairwise_distances_matrix = cdist(X, X)
    a_values = np.zeros(num_points)
    b_values = np.zeros(num_points)
    for i in range(num_clusters):
        mask_i = y_pred == unique_clusters[i]
        mask_i_indices = np.where(mask_i)[0]
        a_values_i = np.sum(pairwise_distances_matrix[mask_i_indices][:, mask_i_indices], axis=1) / np.sum(mask_i)
        a_values[mask_i_indices] = a_values_i
        b_values_i = np.min([
            np.sum(pairwise_distances_matrix[mask_i_indices][:, y_pred == unique_clusters[j]], axis=1) / np.sum(y_pred == unique_clusters[j])
            for j in range(num_clusters) if j != i], axis=0)
        b_values[mask_i_indices] = b_values_i
    results = (b_values - a_values) / np.maximum(a_values, b_values)
    if multi_output:
        return np.round(results, decimal)
    return np.round(np.mean(results), decimal)


def calculate_duda_hart_index(X=None, y_pred=None, decimal=6, raise_error=True, raise_value=np.inf):
    # Find the unique cluster labels
    unique_labels = np.unique(y_pred)
    if len(unique_labels) == 1:
        if raise_error:
            raise ValueError("The Duda-Hart index is undefined when y_pred has only 1 cluster.")
        else:
            return raise_value
    # Compute the pairwise distances between data points
    pairwise_distances = cdist(X, X)
    # Initialize the numerator and denominator for Duda index calculation
    intra_cluster_distances = 0
    inter_cluster_distances = 0
    # Iterate over each unique cluster label
    for label in unique_labels:
        # Find the indices of data points in the current cluster
        cluster_indices = np.where(y_pred == label)[0]
        # Compute the average pairwise distance within the current cluster
        intra_cluster_distances += np.mean(pairwise_distances[np.ix_(cluster_indices, cluster_indices)])
        # Compute the average pairwise distance to other clusters
        other_cluster_indices = np.where(y_pred != label)[0]
        inter_cluster_distances += np.mean(pairwise_distances[np.ix_(cluster_indices, other_cluster_indices)])
    # Calculate the Duda index
    result = intra_cluster_distances / inter_cluster_distances
    return np.round(result, decimal)


def calculate_beale_index(X=None, y_pred=None, decimal=6, raise_error=True, raise_value=np.inf):
    n_clusters = len(np.unique(y_pred))
    if n_clusters == 1:
        if raise_error:
            raise ValueError("The Beale index is undefined when y_pred has only 1 cluster.")
        else:
            return raise_value
    n_samples, n_features = X.shape
    centers, _ = compute_barycenters(X, y_pred)
    sse_within = 0
    sse_between = 0
    for k in range(n_clusters):
        sse_within += np.sum((X[y_pred == k] - centers[k]) ** 2)
        sse_between += np.sum((centers[k] - np.mean(X, axis=0)) ** 2)
    df_within = n_samples - n_clusters
    df_between = n_clusters - 1
    ms_within = sse_within / df_within
    ms_between = sse_between / df_between
    result = ms_within / ms_between
    return np.round(result, decimal)


def calculate_r_squared_index(X=None, y_pred=None, decimal=6):
    n_clusters = len(np.unique(y_pred))
    total_var = np.var(X, axis=0).sum()
    var_within = 0
    for k in range(n_clusters):
        var_within += np.var(X[y_pred == k], axis=0).sum()
    result = (total_var - var_within) / total_var
    return np.round(result, decimal)


def calculate_density_based_clustering_validation_index(X=None, y_pred=None, decimal=6, raise_error=True, raise_value=1.0):
    n_clusters = len(np.unique(y_pred))
    if n_clusters == 1:
        if raise_error:
            raise ValueError("The Density-based Clustering Validation Index is undefined when y_pred has only 1 cluster.")
        else:
            return raise_value
    n_samples, n_features = X.shape
    centroids = np.zeros((n_clusters, n_features))
    for k in range(n_clusters):
        centroids[k] = np.mean(X[y_pred == k], axis=0)
    intra_cluster_distances = cdist(X, centroids, 'euclidean')
    min_inter_cluster_distances = np.zeros(n_samples)
    for i in range(n_samples):
        mask = np.ones(n_samples, dtype=bool)
        mask[i] = False
        mask[y_pred == y_pred[i]] = False
        if np.sum(mask) > 0:
            min_inter_cluster_distances[i] = np.min(cdist(X[i, :].reshape(1, -1), X[mask, :], 'euclidean'))
        else:
            min_inter_cluster_distances[i] = np.inf
    result = np.mean(intra_cluster_distances / np.maximum(min_inter_cluster_distances.reshape(-1, 1), intra_cluster_distances), axis=0).mean()
    return np.round(result, decimal)


def calculate_hartigan_index(X=None, y_pred=None, decimal=6, raise_error=True, raise_value=np.inf):
    centroids, _ = compute_barycenters(X, y_pred)
    num_clusters = len(np.unique(y_pred))
    if num_clusters == 1:
        if raise_error:
            raise ValueError("The Hartigan Index is undefined when y_pred has only 1 cluster.")
        else:
            return raise_value
    hi = 0.0
    for idx in range(num_clusters):
        cluster_data = X[y_pred == idx]
        cluster_centroid = centroids[idx]

        distances_within_cluster = cdist(cluster_data, [cluster_centroid], metric='euclidean') ** 2
        sum_distances_within_cluster = np.sum(distances_within_cluster)

        other_centroids = np.delete(centroids, idx, axis=0)
        closest_other_centroid_index = np.argmin(np.linalg.norm(cluster_centroid - other_centroids, axis=1))
        closest_other_centroid = other_centroids[closest_other_centroid_index]

        distances_to_closest_other_cluster = cdist(cluster_data, [closest_other_centroid], metric='euclidean') ** 2
        sum_distances_to_closest_other_cluster = np.sum(distances_to_closest_other_cluster)

        hi += sum_distances_within_cluster / sum_distances_to_closest_other_cluster
    return np.round(hi, decimal)


def calculate_mutual_info_score(y_true=None, y_pred=None, decimal=6):
    contingency_matrix = compute_contingency_matrix(y_true, y_pred)
    # Convert contingency matrix to probability matrix
    contingency_matrix = contingency_matrix / y_true.shape[0]
    # Calculate marginal probabilities
    cluster_probs_true = np.sum(contingency_matrix, axis=1)
    cluster_probs_pred = np.sum(contingency_matrix, axis=0)
    # Calculate mutual information
    n_clusters_true = len(np.unique(y_true))
    n_clusters_pred = len(np.unique(y_pred))
    mi = 0.0
    for idx in range(n_clusters_true):
        for jdx in range(n_clusters_pred):
            if contingency_matrix[idx, jdx] > 0.0:
                mi += contingency_matrix[idx, jdx] * np.log(contingency_matrix[idx, jdx] / (cluster_probs_true[idx] * cluster_probs_pred[jdx]))
    return np.round(mi, decimal)


def calculate_normalized_mutual_info_score(y_true=None, y_pred=None, decimal=6, raise_error=True, raise_value=0.0):
    mi = calculate_mutual_info_score(y_true, y_pred)
    n_samples = y_true.shape[0]
    n_clusters_true = len(np.unique(y_true))
    n_clusters_pred = len(np.unique(y_pred))
    if n_clusters_true == 1 or n_clusters_pred == 1 or mi == 0:
        # If either of the clusterings has only one cluster, MI is not defined
        if raise_error:
            raise ValueError("The Normalized Mutual Info Score is undefined when MIS = 0 or y_true, y_pred has only 1 cluster.")
        else:
            return raise_value
    # Calculate entropy of true and predicted clusterings
    entropy_true = -np.sum((np.bincount(y_true) / n_samples) * np.log(np.bincount(y_true) / n_samples))
    entropy_pred = -np.sum((np.bincount(y_pred) / n_samples) * np.log(np.bincount(y_pred) / n_samples))
    # Calculate normalized mutual information
    denominator = (entropy_true + entropy_pred) / 2.0
    if denominator == 0:
        return 1.0  # Perfect agreement when both entropies are 0 (all samples in one cluster)
    nmi = mi / denominator
    return np.round(nmi, decimal)


def calculate_rand_score(y_true=None, y_pred=None, decimal=6):
    n_samples = np.int64(y_true.shape[0])
    contingency = compute_contingency_matrix(y_true, y_pred)
    n_c = np.ravel(contingency.sum(axis=1))
    n_k = np.ravel(contingency.sum(axis=0))
    sum_squares = (contingency**2).sum()
    C = np.empty((2, 2), dtype=np.int64)
    C[1, 1] = sum_squares - n_samples
    C[0, 1] = contingency.dot(n_k).sum() - sum_squares
    C[1, 0] = contingency.transpose().dot(n_c).sum() - sum_squares
    C[0, 0] = n_samples**2 - C[0, 1] - C[1, 0] - sum_squares
    numerator = C.diagonal().sum()
    denominator = C.sum()
    if numerator == denominator or denominator == 0:
        # Special limit cases: no clustering since the data is not split; or trivial clustering where each
        # document is assigned a unique cluster. These are perfect matches hence return 1.0.
        return 1.0
    return np.round(numerator / denominator, decimal)


def calculate_adjusted_rand_score(y_true=None, y_pred=None, decimal=6):
    n_samples = np.int64(y_true.shape[0])
    contingency = compute_contingency_matrix(y_true, y_pred)
    n_c = np.ravel(contingency.sum(axis=1))
    n_k = np.ravel(contingency.sum(axis=0))
    sum_squares = (contingency**2).sum()
    C = np.empty((2, 2), dtype=np.int64)
    C[1, 1] = sum_squares - n_samples
    C[0, 1] = contingency.dot(n_k).sum() - sum_squares
    C[1, 0] = contingency.transpose().dot(n_c).sum() - sum_squares
    C[0, 0] = n_samples**2 - C[0, 1] - C[1, 0] - sum_squares
    (tn, fp), (fn, tp) = C
    # convert to Python integer types, to avoid overflow or underflow
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0
    res = 2.0 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    return np.round(res, decimal)


def calculate_fowlkes_mallows_score(y_true=None, y_pred=None, decimal=6, raise_error=True, raise_value=0.0):
    (n_samples,) = y_true.shape
    c = compute_contingency_matrix(y_true, y_pred)
    c = c.astype(np.int64, copy=False)
    tk = np.dot(c.ravel(), c.ravel()) - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples
    if pk == 0.0 or qk == 0.0:
        if raise_error:
            raise ValueError("The Fowlkes Mallows Score is undefined when pk = 0 or qk = 0.")
        else:
            return raise_value
    res = np.sqrt(tk / pk) * np.sqrt(tk / qk)
    return np.round(res, decimal)


def compute_entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return -np.sum(probabilities * np.log2(probabilities))


def compute_conditional_entropy(y_true, y_pred):
    unique_labels_pred = np.unique(y_true)
    entropy_sum = 0
    for label in unique_labels_pred:
        mask = y_pred == label
        cluster_labels_true = y_true[mask]
        cluster_entropy = compute_entropy(cluster_labels_true)
        entropy_sum += len(cluster_labels_true) / len(y_true) * cluster_entropy
    return entropy_sum


def calculate_homogeneity_score(y_true=None, y_pred=None, decimal=6):
    h_labels_true = compute_entropy(y_true)
    h_labels_true_given_pred = compute_conditional_entropy(y_true, y_pred)
    if h_labels_true == 0:
        res = 1.0
    else:
        res = 1. - h_labels_true_given_pred / h_labels_true
    return np.round(res, decimal)


def calculate_completeness_score(y_true=None, y_pred=None, decimal=6):
    return calculate_homogeneity_score(y_pred, y_true, decimal)


def calculate_v_measure_score(y_true=None, y_pred=None, decimal=6):
    h = calculate_homogeneity_score(y_true, y_pred, decimal)
    c = calculate_completeness_score(y_true, y_pred, decimal)
    if h + c == 0:
        res = 0
    else:
        res = 2 * (h * c) / (h + c)
    return np.round(res, decimal)


def calculate_precision_score(y_true=None, y_pred=None, decimal=6):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    return np.round(yy / (yy + ny), decimal)


def calculate_recall_score(y_true=None, y_pred=None, decimal=6):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    return np.round(yy / (yy + yn), decimal)


def calculate_f_measure_score(y_true=None, y_pred=None, decimal=6):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    p = yy / (yy + ny)
    r = yy / (yy + yn)
    return np.round(2 * p * r / (p + r), decimal)


def calculate_czekanowski_dice_score(y_true=None, y_pred=None, decimal=6):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    return np.round(2 * yy / (2 * yy + yn + ny), decimal)


def calculate_hubert_gamma_score(y_true=None, y_pred=None, decimal=6, raise_error=True, raise_value=-1.0):
    n_clusters = len(np.unique(y_pred))
    if n_clusters == 1:
        if raise_error:
            raise ValueError("The Hubert Gamma score is undefined when y_pred has only 1 cluster.")
        else:
            return raise_value
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    NT = yy + yn + ny + nn
    res = (NT*yy - (yy+yn)*(yy+ny)) / np.sqrt((yy+yn)*(yy+ny)*(nn+yn)*(nn+ny))
    return np.round(res, decimal)


def calculate_jaccard_score(y_true=None, y_pred=None, decimal=6):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    return np.round(yy / (yy + yn + ny), decimal)


def calculate_kulczynski_score(y_true=None, y_pred=None, decimal=6):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    res = 0.5 * ((yy / (yy + ny)) + (yy / (yy + yn)))
    return np.round(res, decimal)


def calculate_mc_nemar_score(y_true=None, y_pred=None, decimal=6):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    return np.round((nn - ny) / np.sqrt(nn + ny), decimal)


def calculate_phi_score(y_true=None, y_pred=None, decimal=6, raise_error=True, raise_value=-np.inf):
    n_clusters = len(np.unique(y_pred))
    if n_clusters == 1:
        if raise_error:
            raise ValueError("The Phi score is undefined when y_pred has only 1 cluster.")
        else:
            return raise_value
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    numerator = yy * nn - yn * ny
    denominator = (yy + yn) * (yy + ny) * (yn + nn) * (ny + nn)
    return np.round(numerator / denominator, decimal)


def calculate_rogers_tanimoto_score(y_true=None, y_pred=None, decimal=6):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    cc = (yy + nn) / (yy + nn + 2 * (yn + ny))
    return np.round(cc, decimal)


def calculate_russel_rao_score(y_true=None, y_pred=None, decimal=6):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    NT = yy + yn + ny + nn
    return np.round(yy / NT, decimal)


def calculate_sokal_sneath1_score(y_true=None, y_pred=None, decimal=6):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    cc = yy / (yy + 2 * (yn + ny))
    return np.round(cc, decimal)


def calculate_sokal_sneath2_score(y_true=None, y_pred=None, decimal=6):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    cc = (yy + nn) / (yy + nn + 0.5 * (yn + ny))
    return np.round(cc, decimal)


def calculate_purity_score(y_true=None, y_pred=None, decimal=6):
    # Find the number of data points
    N = len(y_true)
    # Find the unique class labels in the true labels
    unique_classes = np.unique(y_true)
    # Initialize the purity score
    purity = 0
    # Iterate over each unique class label
    for c in unique_classes:
        # Find the indices of data points with the current class label in the true labels
        class_indices = np.where(y_true == c)[0]
        # Find the corresponding predicted labels for these data points
        class_predictions = y_pred[class_indices]
        # Count the occurrences of each predicted label
        class_predictions = np.round(class_predictions).astype(int)
        class_counts = np.bincount(class_predictions)
        # Add the size of the majority class to the purity score
        purity += np.max(class_counts)
    # Normalize the purity score by dividing by the total number of data points
    return np.round(purity / N, decimal)


def calculate_entropy_score(y_true=None, y_pred=None, decimal=6):
    # Find the number of data points
    N = len(y_true)
    # Find the unique class labels in the true labels
    unique_classes = np.unique(y_true)
    result = 0
    # Iterate over each unique class label
    for c in unique_classes:
        # Find the indices of data points with the current class label in the true labels
        class_indices = np.where(y_true == c)[0]
        # Find the corresponding predicted labels for these data points
        class_predictions = y_pred[class_indices]
        class_predictions = np.round(class_predictions).astype(int)
        # Count the occurrences of each predicted label
        class_counts = np.bincount(class_predictions)
        # Normalize the class counts by dividing by the total number of data points in the cluster
        class_distribution = class_counts / len(class_predictions)
        # Compute the entropy of the cluster
        cluster_entropy = calculate_entropy(class_distribution, base=2)
        # Weight the entropy by the relative size of the cluster
        cluster_size = len(class_indices)
        result += (cluster_size / N) * cluster_entropy
    return np.round(result, decimal)


def compute_nd_splus_sminus_t(y_true=None, y_pred=None):
    """concordant_discordant"""
    n_samples = len(y_true)
    nd = n_samples * (n_samples - 1) / 2
    s_plus = 0.  # Number of concordant comparisons
    t = 0.  # Number of comparisons of two pairs of objects with same cluster labels
    for idx in range(n_samples - 1):
        t += np.sum((y_true[idx] == y_true[idx + 1:]) & (y_pred[idx] == y_pred[idx + 1:]))
        s_plus += np.sum((y_true[idx] == y_true[idx + 1:]) & (y_pred[idx] == y_pred[idx + 1:]))
        s_plus += np.sum((y_true[idx] != y_true[idx + 1:]) & (y_pred[idx] != y_pred[idx + 1:]))
    s_minus = nd - s_plus       # Number of discordant comparisons
    return nd, s_plus, s_minus, t


def calculate_tau_score(y_true=None, y_pred=None, decimal=6):
    """
    Cluster Validation for Mixed-Type Data: Paper
    """
    nd, s_plus, s_minus, t = compute_nd_splus_sminus_t(y_true, y_pred)
    res = (s_plus - s_minus) / np.sqrt((nd - t) * nd)
    return np.round(res, decimal)


def calculate_gamma_score(y_true=None, y_pred=None, decimal=6):
    """
    Cluster Validation for Mixed-Type Data: Paper
    """
    nd, s_plus, s_minus, t = compute_nd_splus_sminus_t(y_true, y_pred)
    res = (s_plus - s_minus) / (s_plus + s_minus)
    return np.round(res, decimal)


def calculate_gplus_score(y_true=None, y_pred=None, decimal=6):
    """
    Cluster Validation for Mixed-Type Data: Paper
    """
    nd, s_plus, s_minus, t = compute_nd_splus_sminus_t(y_true, y_pred)
    res = s_minus / nd
    return np.round(res, decimal)
