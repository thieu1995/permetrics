#!/usr/bin/env python
# Created by "Thieu" at 23:23, 28/06/2026 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import coo_matrix


def _raise_err(metric, reason):
    raise ValueError(f"The {metric} score is undefined because {reason}.")


def sum_comb(x):
    """
    Calculate the total number of pairs nC2 = n(n-1)/2
    """
    return np.sum(x * (x - 1) / 2)


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

    # # Calculate the between-cluster variance
    # between_var = 0.0
    # for i in range(n_clusters):
    #     cluster_mask = labels == i
    #     cluster_size = np.sum(cluster_mask)
    #     cluster_mean = np.mean(X[cluster_mask], axis=0)
    #     between_var += cluster_size * np.sum((cluster_mean - overall_mean) ** 2)
    # return between_var

    # Calculate between-cluster variance and cluster sizes
    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    cluster_means = np.array([np.mean(X[labels == i], axis=0) for i in range(n_clusters)])
    between_var = np.sum(cluster_sizes * np.sum((cluster_means - overall_mean) ** 2, axis=1))
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
    """
    contingency = compute_contingency_matrix(y_true, y_pred)
    n = len(y_true)
    yy = sum_comb(contingency)

    sum_rows = np.ravel(contingency.sum(axis=1))
    sum_cols = np.ravel(contingency.sum(axis=0))

    yy_plus_yn = sum_comb(sum_rows)  # same cluster in y_true
    yy_plus_ny = sum_comb(sum_cols)  # same cluster in y_pred

    yn = yy_plus_yn - yy
    ny = yy_plus_ny - yy

    total_pairs = n * (n - 1) / 2
    nn = total_pairs - (yy + yn + ny)

    res = np.array([yy, yn, ny, nn], dtype=np.float64)
    if normalize:
        return res / np.sum(res)
    return res


def calculate_sum_squared_error_index(X=None, y_pred=None):
    centers, _ = compute_barycenters(X, y_pred)
    centroid_distances = centers[y_pred]
    squared_distances = np.sum((X - centroid_distances) ** 2, axis=1)
    return np.sum(squared_distances)


def calculate_mean_squared_error_index(X=None, y_pred=None):
    centers, _ = compute_barycenters(X, y_pred)
    centroid_distances = centers[y_pred]
    squared_distances = np.sum((X - centroid_distances) ** 2, axis=1)
    return np.mean(squared_distances)


def calculate_ball_hall_index(X=None, y_pred=None):
    pred_labels = np.unique(y_pred)
    wgss = 0
    ## For each cluster, find the centroid and then the within-group SSE
    for k in pred_labels:
        centroid_mask = y_pred == k
        cluster_k = X[centroid_mask]
        centroid = np.mean(cluster_k, axis=0)
        wgss += np.sum((cluster_k - centroid) ** 2)
    return wgss / len(pred_labels)


def calculate_calinski_harabasz_index(X=None, y_pred=None, force_finite=True, finite_value=0.0):
    """
    Calinski-Harabasz Index (Variance Ratio Criterion).
    """
    labels = np.asarray(y_pred)
    n_samples, _ = X.shape
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Edge-case 1: single_cluster (k = 1) -> k - 1 = 0
    if n_clusters == 1:
        if force_finite:
            return float(finite_value)
        raise ValueError("The Calinski-Harabasz index is undefined when there is only 1 cluster.")

    # Edge-case 2: all_singletons (k = N) -> N - k = 0
    if n_clusters == n_samples:
        if force_finite:
            return float(finite_value)
        raise ValueError("The Calinski-Harabasz index is undefined when all samples are singletons.")

    extra_disp, intra_disp = 0.0, 0.0
    mean = np.mean(X, axis=0)

    for k in unique_labels:
        cluster_k = X[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
        intra_disp += np.sum((cluster_k - mean_k) ** 2)

    # Edge-case 3: zero_variance_data (SS_W = 0)
    if intra_disp == 0.0:
        if force_finite:
            return float(finite_value)
        raise ValueError("The Calinski-Harabasz index is undefined when within-cluster dispersion is strictly 0.")

    ch_score = (extra_disp / (n_clusters - 1)) / (intra_disp / (n_samples - n_clusters))
    return float(ch_score)


def calculate_xie_beni_index(X=None, y_pred=None, force_finite=True, finite_value=1e10):
    unique_labels = np.unique(y_pred)
    n_clusters = len(unique_labels)

    # Edge-case 1: Single cluster
    if n_clusters == 1:
        if force_finite:
            return float(finite_value)
        raise ValueError("The Xie-Beni index is undefined when y_pred has only 1 cluster.")

    centroids, _ = compute_barycenters(X, y_pred)
    # Within-group sum of squares
    wgss = np.sum(np.min(cdist(X, centroids, metric='euclidean'), axis=1) ** 2)
    MinSqDist = np.min(pdist(centroids, metric='sqeuclidean'))

    # Edge-case 2: zero_variance_data
    if MinSqDist == 0.0:
        if force_finite:
            return float(finite_value)
        raise ValueError("The Xie-Beni index is undefined when the minimum distance between centroids is 0.")

    res = (wgss / X.shape[0]) / MinSqDist
    if np.isnan(res) or np.isinf(res):
        if force_finite:
            return float(finite_value)
        raise ValueError("XBI calculation resulted in NaN/Inf.")
    return float(res)


def calculate_banfeld_raftery_index(X=None, y_pred=None, force_finite=True, finite_value=1e10):
    clusters_dict, cluster_sizes_dict = compute_clusters(y_pred)
    res = 0.0
    for k in clusters_dict.keys():
        X_k = X[clusters_dict[k]]
        cluster_dispersion = np.trace(compute_WG(X_k)) / cluster_sizes_dict[k]
        if cluster_sizes_dict[k] == 1:
            if force_finite:
                return finite_value
            else:
                raise ValueError("The Banfeld-Raftery index is undefined when at least 1 cluster has only 1 sample.")
        if cluster_sizes_dict[k] > 1:
            res += cluster_sizes_dict[k] * np.log(cluster_dispersion)
    return res


def calculate_davies_bouldin_index(X=None, y_pred=None, force_finite=True, finite_value=1e10):
    clusters_dict, cluster_sizes_dict = compute_clusters(y_pred)
    centers, _ = compute_barycenters(X, y_pred)
    n_clusters = len(clusters_dict)
    if n_clusters == 1:
        if force_finite:
            return finite_value
        else:
            raise ValueError("The Davies-Bouldin index is undefined when y_pred has only 1 cluster.")
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
    return cc / n_clusters


def calculate_davies_bouldin_index(X=None, y_pred=None, force_finite=True, finite_value=1e10):
    unique_labels = np.unique(y_pred)
    n_clusters = len(unique_labels)

    # Edge-case 1
    if n_clusters == 1:
        if force_finite:
            return float(finite_value)
        raise ValueError("The Davies-Bouldin index is undefined when y_pred has only 1 cluster.")

    centers = np.empty((n_clusters, X.shape[1]))
    deltas = np.empty(n_clusters)

    for i, k in enumerate(unique_labels):
        X_k = X[y_pred == k]
        center_k = np.mean(X_k, axis=0)
        centers[i] = center_k
        deltas[i] = np.mean(np.linalg.norm(X_k - center_k, axis=1))
    center_dists = squareform(pdist(centers, metric='euclidean'))

    # Edge-case 2
    np.fill_diagonal(center_dists, np.inf)
    if np.any(center_dists == 0.0):
        if force_finite:
            return float(finite_value)
        raise ValueError("Davies-Bouldin index is undefined when two clusters have identical centers (distance is 0).")

    delta_sums = deltas[:, None] + deltas[None, :]
    ratios = delta_sums / center_dists
    max_ratios = np.max(ratios, axis=1)
    res = np.mean(max_ratios)
    if np.isnan(res) or np.isinf(res):
        if force_finite:
            return float(finite_value)
        raise ValueError("DBI calculation resulted in NaN/Inf.")
    return float(res)

def calculate_det_ratio_index(X=None, y_pred=None, force_finite=True, finite_value=-1e10):
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
        if force_finite:
            return finite_value
        else:
            raise ValueError("The Det-Ratio index is undefined when determinant of matrix is 0.")
    return np.linalg.det(T) / t1


def calculate_dunn_index(X=None, y_pred=None, use_modified=True, force_finite=True, finite_value=0.0):
    centers, _ = compute_barycenters(X, y_pred)
    unique_labels = np.unique(y_pred)
    n_clusters = len(unique_labels)

    if n_clusters == 1:
        if force_finite:
            return float(finite_value)
        raise ValueError("The Dunn index is undefined when y_pred has only 1 cluster.")

    # Calculate dmin (Inter-cluster separation)
    dmin = np.inf
    if use_modified:
        for k0 in range(n_clusters - 1):
            for k1 in range(k0 + 1, n_clusters):
                points = X[y_pred == unique_labels[k1]]
                dkk = np.min(cdist(points, centers[k0].reshape(1, -1), metric='euclidean'))
                dmin = min(dmin, dkk)
    else:
        for k0 in range(n_clusters - 1):
            for k1 in range(k0 + 1, n_clusters):
                points1 = X[y_pred == unique_labels[k0]]
                points2 = X[y_pred == unique_labels[k1]]
                dkk = np.min(cdist(points1, points2, metric='euclidean'))
                dmin = min(dmin, dkk)

    # Calculate dmax (Maximum intra-cluster diameter)
    dmax = 0.0
    for k_label in unique_labels:
        points = X[y_pred == k_label]
        if len(points) > 1:
            dk = np.max(pdist(points, metric="euclidean"))
            dmax = max(dmax, dk)

    if dmax == 0.0:
        if force_finite:
            return float(finite_value)
        raise ValueError("The Dunn index is undefined when the maximum intra-cluster distance is 0.")
    return float(dmin / dmax)


def calculate_ksq_detw_index(X=None, y_pred=None, use_normalized=True, force_finite=True, finite_value=0.0):
    unique_labels = np.unique(y_pred)
    n_clusters = len(unique_labels)

    if n_clusters == 1:
        if force_finite:
            return float(finite_value)
        raise ValueError("The KDI metric is undefined when there is only 1 cluster.")

    scatter_matrices = np.zeros((X.shape[1], X.shape[1]))  # shape of (n_features, n_features)
    for k_label in unique_labels:
        X_k = X[y_pred == k_label]
        if len(X_k) > 1:
            scatter_matrices += compute_WG(X_k)

    if use_normalized:
        mat_min = np.min(scatter_matrices)
        mat_max = np.max(scatter_matrices)
        mat_range = mat_max - mat_min
        if mat_range == 0.0:
            if force_finite:
                return float(finite_value)
            raise ValueError("KDI normalization failed: internal scatter matrices have zero variance.")
        scatter_matrices = (scatter_matrices - mat_min) / mat_range

    res = (n_clusters ** 2) * np.linalg.det(scatter_matrices)
    if np.isnan(res) or np.isinf(res):
        if force_finite:
            return float(finite_value)
        raise ValueError("KDI calculation resulted in NaN/Inf due to matrix determinant instability.")
    return float(res)


def calculate_log_det_ratio_index(X=None, y_pred=None, force_finite=True, finite_value=-1e10):
    clusters_dict, cluster_sizes_dict = compute_clusters(y_pred)
    centers, _ = compute_barycenters(X, y_pred)
    T = compute_WG(X)
    WG = np.zeros((X.shape[1], X.shape[1]))  # shape of (n_features, n_features)
    for label, indices in clusters_dict.items():
        X_k = X[indices]
        WG += compute_WG(X_k)
    t2 = np.linalg.det(WG)
    if t2 == 0:
        if force_finite:
            return finite_value
        else:
            raise ValueError("The Log Det Ratio Index is undefined when determinant of matrix WG is 0.")
    t1 = np.linalg.det(T) / t2
    if t1 <= 0:
        if force_finite:
            return finite_value
        else:
            raise ValueError("The Log Det Ratio Index is undefined when det(T)/det(WG) <= 0.")
    return X.shape[0] * np.log(t1)


def calculate_silhouette_index(X, y_pred, chunk_size=5000, multi_output=False, force_finite=True, finite_value=-1.0):
    """
    A chunk-based implementation of Silhouette Score to prevent OOM on large datasets (100K+).
    """
    unique_labels = np.unique(y_pred)
    if len(unique_labels) == 1:
        if force_finite:
            return finite_value if not multi_output else np.full(len(X), finite_value)
        else:
            raise ValueError("The Silhouette Index is undefined when y_pred has only 1 cluster.")

    n_samples = len(X)
    silhouette_scores = np.zeros(n_samples)

    # Chunk processing is used to prevent RAM overflow.
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        X_chunk = X[start:end]
        y_chunk = y_pred[start:end]

        # Calculate the distance from the current chunk to ALL other points, matrix shape (chunk_size, n_samples)
        D_chunk = cdist(X_chunk, X, metric='euclidean')

        a_chunk = np.zeros(len(X_chunk))
        b_chunk = np.full(len(X_chunk), np.inf)

        for label in unique_labels:
            mask_label = (y_pred == label)
            N_label = np.sum(mask_label)
            if N_label == 0:
                continue

            # Total distance from the chunk to the `label` cluster
            sum_D_to_label = np.sum(D_chunk[:, mask_label], axis=1)
            in_label_mask = (y_chunk == label)

            # 1. Calculate a(i) - Same cluster
            if np.any(in_label_mask):
                denom = N_label - 1
                if denom > 0:
                    a_chunk[in_label_mask] = sum_D_to_label[in_label_mask] / denom
                else:
                    a_chunk[in_label_mask] = 0.0  # By convention, a_i = 0 if the cluster has only 1 element.

            # 2. Calculate b(i) - Different cluster
            out_label_mask = ~in_label_mask
            if np.any(out_label_mask):
                mean_D_to_label = sum_D_to_label[out_label_mask] / N_label
                b_chunk[out_label_mask] = np.minimum(b_chunk[out_label_mask], mean_D_to_label)

        # 3. Calculate the Silhouette coefficient for the current chunk.
        valid = (a_chunk != 0) | (b_chunk != np.inf)
        s_chunk = np.zeros(len(X_chunk))
        s_chunk[valid] = (b_chunk[valid] - a_chunk[valid]) / np.maximum(a_chunk[valid], b_chunk[valid])

        # If the cluster has only 1 point, silhouette = 0
        s_chunk[a_chunk == 0.0] = 0.0
        silhouette_scores[start:end] = s_chunk

    if multi_output:
        return silhouette_scores
    return np.mean(silhouette_scores)


def calculate_duda_hart_index(X=None, y_pred=None, chunk_size=5000, force_finite=True, finite_value=1e10):
    unique_labels = np.unique(y_pred)
    if len(unique_labels) == 1:
        if force_finite:
            return finite_value
        else:
            raise ValueError("The Duda-Hart index is undefined when y_pred has only 1 cluster.")
    n_samples = len(X)
    # Initialize a dict to store the total distance and cluster size to avoid errors if labels are missing (not contiguous).
    intra_sums = {label: 0.0 for label in unique_labels}
    inter_sums = {label: 0.0 for label in unique_labels}
    cluster_sizes = {label: np.sum(y_pred == label) for label in unique_labels}

    # Batch processing prevents RAM overflow.
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        X_chunk = X[start:end]
        y_chunk = y_pred[start:end]

        # Matrix shape (chunk_size, n_samples)
        D_chunk = cdist(X_chunk, X, metric='euclidean')
        for label in unique_labels:
            row_mask = (y_chunk == label)
            if not np.any(row_mask):
                continue

            col_mask_intra = (y_pred == label)
            col_mask_inter = (y_pred != label)
            # Sum the total intra and inter distances for the 'label' cluster.
            intra_sums[label] += np.sum(D_chunk[np.ix_(row_mask, col_mask_intra)])
            inter_sums[label] += np.sum(D_chunk[np.ix_(row_mask, col_mask_inter)])

    # Calculate the mean from the sums of the accumulated totals.
    intra_cluster_distances = 0.0
    inter_cluster_distances = 0.0

    for label in unique_labels:
        n_k = cluster_sizes[label]
        if n_k > 0:
            # The np.mean of the N*N sub-matrix is equivalent to Sum / N^2
            intra_cluster_distances += intra_sums[label] / (n_k ** 2)

            n_other = n_samples - n_k
            if n_other > 0:
                # np.mean of sub-matrix N*M is equivalent to Sum / (N*M)
                inter_cluster_distances += inter_sums[label] / (n_k * n_other)

    # Handle cases where the denominator is zero to avoid ZeroDivisionError.
    if inter_cluster_distances == 0:
        return finite_value if force_finite else np.inf
    return intra_cluster_distances / inter_cluster_distances


def calculate_beale_index(X=None, y_pred=None, force_finite=True, finite_value=1e10):
    """
    Beale Index (BI).
    """
    pred_labels = np.unique(y_pred)
    n_clusters = len(pred_labels)
    n_samples, n_features = X.shape

    # Edge-case 1: Single cluster
    if n_clusters == 1:
        if force_finite:
            return float(finite_value)
        raise ValueError("The Beale index is undefined when y_pred has only 1 cluster.")

    # Edge-case 2: All singletons
    if n_samples == n_clusters:
        if force_finite:
            return float(finite_value)
        raise ValueError("The Beale index is undefined when all samples are singletons.")

    sse_within = 0.0
    sse_between = 0.0
    global_mean = np.mean(X, axis=0)

    for k in pred_labels:
        cluster_points = X[y_pred == k]
        c_k = np.mean(cluster_points, axis=0)
        sse_within += np.sum((cluster_points - c_k) ** 2)
        sse_between += np.sum((c_k - global_mean) ** 2)

    df_within = n_samples - n_clusters
    df_between = n_clusters - 1
    ms_within = sse_within / df_within
    ms_between = sse_between / df_between

    # Edge-case 3: Zero variance data
    if ms_between == 0.0:
        if force_finite:
            return float(finite_value)
        raise ValueError("The Beale index is undefined when between-cluster variance is strictly 0.")

    result = ms_within / ms_between
    if np.isnan(result) or np.isinf(result):
        if force_finite:
            return float(finite_value)
        raise ValueError("Beale index calculation resulted in NaN/Inf.")
    return float(result)


def calculate_r_squared_index(X=None, y_pred=None):
    tss = compute_TSS(X)
    return (tss - compute_WGSS(X, y_pred)) / tss if tss > 0 else 1.0


def calculate_dbcv_score(X=None, y_pred=None, force_finite=True, finite_value=0.0):
    """
    Density-Based Clustering Validation (DBCV) - Moulavi et al. (2014).
    Valid for arbitrarily shaped clusters. Range: [-1, 1].
    """
    if X is None or y_pred is None:
        raise ValueError("Both X and y_pred must be provided for DBCV.")

    labels = np.asarray(y_pred)
    unique_classes = np.unique(labels)
    # Ignore noise (commonly labeled as -1 in density algorithms like DBSCAN)
    clusters = unique_classes[unique_classes != -1]

    n_clusters = len(clusters)
    if n_clusters < 2:
        if force_finite:
            return finite_value
        else:
            raise ValueError("The DBCV score requires at least 2 valid clusters to compute separation.")

    n_samples, d = X.shape
    cluster_indices = [np.where(labels == c)[0] for c in clusters]

    # 1. Compute All-points core distance
    core_distances = np.zeros(n_samples)
    for idxs in cluster_indices:
        n_i = len(idxs)
        if n_i > 1:
            # pdist avoids an NxN matrix for the whole dataset, saving memory per cluster
            D_cluster = squareform(pdist(X[idxs], metric='euclidean'))
            core_distances[idxs] = (np.sum(D_cluster ** d, axis=1) / (n_i - 1)) ** (1.0 / d)

    # 2. Extract valid clustered points to build the Mutual Reachability Distance (MRD) graph
    valid_mask = labels != -1
    X_valid = X[valid_mask]
    core_valid = core_distances[valid_mask]

    # Compute base pairwise distances among clustered points
    D_full_valid = squareform(pdist(X_valid, metric='euclidean'))

    # Apply MRD Formula
    MRD = np.maximum(D_full_valid, core_valid[:, None])
    MRD = np.maximum(MRD, core_valid[None, :])

    # Map old global indices to new valid-only indices
    cluster_valid_indices = [np.where(labels[valid_mask] == c)[0] for c in clusters]

    # 3. Compute Density Sparseness (DSC) via MST
    dsc = np.zeros(n_clusters)
    mst_graphs = []

    for i, idxs in enumerate(cluster_valid_indices):
        if len(idxs) > 1:
            sub_mrd = MRD[np.ix_(idxs, idxs)]
            # Construct Minimum Spanning Tree
            mst = minimum_spanning_tree(sub_mrd).toarray()
            mst = mst + mst.T  # Symmetrize
            mst_graphs.append(mst)

            # Filter internal nodes (degree > 1)
            degrees = np.sum(mst > 0, axis=0)
            internal_nodes = degrees > 1

            if np.any(internal_nodes):
                internal_edges = mst[np.ix_(internal_nodes, internal_nodes)]
                dsc[i] = np.max(internal_edges) if np.any(internal_edges > 0) else np.max(mst)
            else:
                dsc[i] = np.max(mst)
        else:
            mst_graphs.append(np.array([[0.0]]))
            dsc[i] = 0.0

    # 4. Compute Density Separation (DSPC)
    dspc = np.full((n_clusters, n_clusters), np.inf)
    for i in range(n_clusters):
        nodes_i = cluster_valid_indices[i]
        deg_i = np.sum(mst_graphs[i] > 0, axis=0) if len(nodes_i) > 1 else np.array([1])
        in_i = nodes_i[deg_i > 1] if np.any(deg_i > 1) else nodes_i

        for j in range(i + 1, n_clusters):
            nodes_j = cluster_valid_indices[j]
            deg_j = np.sum(mst_graphs[j] > 0, axis=0) if len(nodes_j) > 1 else np.array([1])
            in_j = nodes_j[deg_j > 1] if np.any(deg_j > 1) else nodes_j

            if len(in_i) > 0 and len(in_j) > 0:
                sep = np.min(MRD[np.ix_(in_i, in_j)])
                dspc[i, j] = dspc[j, i] = sep

    # 5. Calculate Validity Index of each Cluster
    min_dspc = np.min(dspc, axis=1)
    v_clusters = np.zeros(n_clusters)

    for i in range(n_clusters):
        if np.isinf(min_dspc[i]) and dsc[i] == 0:
            v_clusters[i] = 0.0
        else:
            v_clusters[i] = (min_dspc[i] - dsc[i]) / max(min_dspc[i], dsc[i])

    # 6. Global DBCV Index
    # The sum of weights implicitly penalizes noise because n_samples includes noise
    weights = np.array([len(idxs) for idxs in cluster_indices]) / n_samples
    return np.sum(weights * v_clusters), dict(zip(clusters, v_clusters))


def calculate_hartigan_index(X=None, y_pred=None, force_finite=True, finite_value=1e10):
    unique_labels = np.unique(y_pred)
    num_clusters = len(unique_labels)
    if num_clusters == 1:
        if force_finite:
            return float(finite_value)
        raise ValueError("The Hartigan Index is undefined when y_pred has only 1 cluster.")

    centroids, _ = compute_barycenters(X, y_pred)
    hi = 0.0
    for i, label in enumerate(unique_labels):
        cluster_data = X[y_pred == label]
        cluster_centroid = centroids[i]
        distances_within = cdist(cluster_data, [cluster_centroid], metric='euclidean') ** 2
        sum_dist_within = np.sum(distances_within)
        other_centroids = np.delete(centroids, i, axis=0)

        # Determine the closest alternative centroid
        closest_idx = np.argmin(np.linalg.norm(cluster_centroid - other_centroids, axis=1))
        closest_centroid = other_centroids[closest_idx]

        distances_to_other = cdist(cluster_data, [closest_centroid], metric='euclidean') ** 2
        sum_dist_to_other = np.sum(distances_to_other)

        # Zero-division protection: Handle the zero variance edge-case
        if sum_dist_to_other == 0.0:
            if force_finite:
                return float(finite_value)
            raise ValueError(f"Hartigan Index calculation failed: Cluster {label} has zero variance to its closest neighbor.")
        hi += sum_dist_within / sum_dist_to_other

    # Final safety catch for floating-point overflow bounds
    if np.isnan(hi) or np.isinf(hi):
        if force_finite:
            return float(finite_value)
        raise ValueError("Hartigan Index calculation resulted in NaN/Inf.")
    return float(hi)


def calculate_mutual_info_score(y_true=None, y_pred=None):
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
    return mi


def calculate_normalized_mutual_info_score(y_true=None, y_pred=None, force_finite=True, finite_value=0.0):
    mi = calculate_mutual_info_score(y_true, y_pred)
    n_samples = y_true.shape[0]
    n_clusters_true = len(np.unique(y_true))
    n_clusters_pred = len(np.unique(y_pred))
    if n_clusters_true == 1 or n_clusters_pred == 1 or mi == 0:
        # If either of the clusterings has only one cluster, MI is not defined
        if force_finite:
            return finite_value
        else:
            raise ValueError("The Normalized Mutual Info Score is undefined when MIS = 0 or y_true, y_pred has only 1 cluster.")
    # Calculate entropy of true and predicted clusterings
    entropy_true = -np.sum((np.bincount(y_true) / n_samples) * np.log(np.bincount(y_true) / n_samples))
    entropy_pred = -np.sum((np.bincount(y_pred) / n_samples) * np.log(np.bincount(y_pred) / n_samples))
    # Calculate normalized mutual information
    denominator = (entropy_true + entropy_pred) / 2.0
    if denominator == 0:
        return 1.0  # Perfect agreement when both entropies are 0 (all samples in one cluster)
    nmi = mi / denominator
    return nmi


def calculate_rand_score(y_true=None, y_pred=None):
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
    return numerator / denominator


def calculate_adjusted_rand_score(y_true=None, y_pred=None):
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
    return res


def calculate_fowlkes_mallows_score(y_true=None, y_pred=None, force_finite=True, finite_value=0.0):
    (n_samples,) = y_true.shape
    c = compute_contingency_matrix(y_true, y_pred)
    c = c.astype(np.int64, copy=False)
    tk = np.dot(c.ravel(), c.ravel()) - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples
    if pk == 0.0 or qk == 0.0:
        if force_finite:
            return finite_value
        else:
            raise ValueError("The Fowlkes Mallows Score is undefined when pk = 0 or qk = 0.")
    res = np.sqrt(tk / pk) * np.sqrt(tk / qk)
    return res


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


def calculate_homogeneity_score(y_true=None, y_pred=None, force_finite=True, finite_value=1.0):
    if len(np.unique(y_pred)) == 1:
        if force_finite:
            return finite_value
        else:
            raise ValueError("The Homogeneity Score is undefined when y_pred has only 1 cluster.")

    h_labels_true = compute_entropy(y_true)
    h_labels_true_given_pred = compute_conditional_entropy(y_true, y_pred)
    if h_labels_true == 0:
        res = 1.0
    else:
        res = 1. - h_labels_true_given_pred / h_labels_true
    return res


def calculate_completeness_score(y_true=None, y_pred=None, force_finite=True, finite_value=1.0):
    return calculate_homogeneity_score(y_pred, y_true, force_finite, finite_value)


def calculate_v_measure_score(y_true=None, y_pred=None, beta=1.0, force_finite=True, finite_value=1.0):
    h = calculate_homogeneity_score(y_true, y_pred, force_finite, finite_value)
    c = calculate_completeness_score(y_true, y_pred, force_finite, finite_value)
    # Handle the boundary case where both homogeneity and completeness are zero
    if h + c == 0.0:
        res = 0.0
    else:
        res = ((1.0 + beta) * h * c) / (beta * h + c)
    return float(res)


def calculate_precision_score(y_true=None, y_pred=None, force_finite=True, finite_value=1.0):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    den = yy + ny
    if den == 0:
        return finite_value if force_finite else _raise_err("Precision", "y_pred contains only singletons")
    return yy / den


def calculate_recall_score(y_true=None, y_pred=None, force_finite=True, finite_value=1.0):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    den = yy + yn
    if den == 0:
        return finite_value if force_finite else _raise_err("Recall", "y_true contains only singletons")
    return yy / den


def calculate_f_measure_score(y_true=None, y_pred=None, beta=1.0, force_finite=True, finite_value=1.0):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    b2 = beta ** 2
    den = (1 + b2) * yy + b2 * yn + ny
    if den == 0:
        return finite_value if force_finite else _raise_err("F-Measure", "both Precision and Recall are undefined/zero")
    return ((1 + b2) * yy) / den


def calculate_czekanowski_dice_score(y_true=None, y_pred=None, force_finite=True, finite_value=1.0):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    den = 2 * yy + yn + ny
    if den == 0:
        return finite_value if force_finite else _raise_err("CDS", "no positive co-clusterings exist")
    return (2 * yy) / den


def calculate_hubert_gamma_score(y_true=None, y_pred=None, force_finite=True, finite_value=0.0):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    den_sq = (yy + yn) * (yy + ny) * (nn + yn) * (nn + ny)
    if den_sq == 0:
        return finite_value if force_finite else _raise_err("Hubert Gamma", "indicator variables have zero variance")
    NT = yy + yn + ny + nn
    return (NT * yy - (yy + yn) * (yy + ny)) / np.sqrt(den_sq)


def calculate_jaccard_score(y_true=None, y_pred=None, force_finite=True, finite_value=0.0):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    den = yy + yn + ny
    if den == 0:
        return finite_value if force_finite else _raise_err("Jaccard", "no positive co-clusterings exist")
    return yy / den


def calculate_kulczynski_score(y_true=None, y_pred=None, force_finite=True, finite_value=0.0):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    den_p = yy + ny
    den_r = yy + yn
    if den_p == 0 or den_r == 0:
        return finite_value if force_finite else _raise_err("Kulczynski", "Precision or Recall denominator is zero")
    return 0.5 * ((yy / den_p) + (yy / den_r))


def calculate_mc_nemar_score(y_true=None, y_pred=None, force_finite=True, finite_value=0.0):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    den = nn + ny
    if den == 0:
        return finite_value if force_finite else _raise_err("McNemar", "denominator (nn + ny) is zero")
    return (nn - ny) / np.sqrt(den)


def calculate_phi_score(y_true=None, y_pred=None, force_finite=True, finite_value=0.0):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    den_sq = (yy + yn) * (yy + ny) * (yn + nn) * (ny + nn)
    if den_sq == 0:
        return finite_value if force_finite else _raise_err("Phi", "indicator variables have zero variance")
    return (yy * nn - yn * ny) / np.sqrt(den_sq)


def calculate_rogers_tanimoto_score(y_true=None, y_pred=None, force_finite=True, finite_value=0.0):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    den_sq = yy + nn + 2 * (yn + ny)
    if den_sq == 0:
        return finite_value if force_finite else _raise_err("Rogers Tanimoto", "denominator is zero")
    return (yy + nn) / den_sq


def calculate_russel_rao_score(y_true=None, y_pred=None, force_finite=True, finite_value=0.0):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    NT = yy + yn + ny + nn
    if NT == 0:
        return finite_value if force_finite else _raise_err("Russel Rao", "denominator is zero")
    return yy / NT


def calculate_sokal_sneath1_score(y_true=None, y_pred=None, force_finite=True, finite_value=0.0):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    den = yy + 2 * (yn + ny)
    if den == 0:
        return finite_value if force_finite else _raise_err("Sokal Sneath 1", "no positive co-clusterings exist")
    return yy / den


def calculate_sokal_sneath2_score(y_true=None, y_pred=None, force_finite=True, finite_value=0.0):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    den = yy + nn + 0.5 * (yn + ny)
    if den == 0:
        return finite_value if force_finite else _raise_err("Sokal Sneath 2", "dataset has fewer than 2 samples")
    return (yy + nn) / den


def calculate_purity_score(y_true=None, y_pred=None):
    """
    O(N) Vectorized Purity Score. Safe for arbitrary label formats.
    """
    contingency = compute_contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency, axis=0)) / len(y_true)


def calculate_entropy_score(y_true=None, y_pred=None):
    """
    O(N) Vectorized Cluster Entropy Score (Corrected Formulation).
    """
    N = len(y_true)
    if N == 0:
        return 0.0
    contingency = compute_contingency_matrix(y_true, y_pred)
    cluster_sizes = np.sum(contingency, axis=0)

    # Remove empty cluster
    non_empty = cluster_sizes > 0
    contingency = contingency[:, non_empty]
    cluster_sizes = cluster_sizes[non_empty]

    probs = contingency / cluster_sizes  # Distribute y_true inside each y_pred
    probs = probs[probs > 0]  # Avoid log(0)

    # Calculate entropy of each cluster and multiply size of cluster
    cluster_entropies = -np.sum(probs * np.log2(probs), axis=0)
    return np.sum((cluster_sizes / N) * cluster_entropies)


def compute_nd_splus_sminus_t(y_true, y_pred):
    """
    Optimized version computing Concordant/Discordant pairs in O(N) using Combinatorics.
    """
    n = len(y_true)
    nd = n * (n - 1) // 2  # Total pairs
    contingency = compute_contingency_matrix(y_true, y_pred)

    a = int(sum_comb(contingency)) # Same True, Same Pred (it is t)
    sum_rows = np.ravel(contingency.sum(axis=1))
    sum_cols = np.ravel(contingency.sum(axis=0))

    same_true = int(sum_comb(sum_rows))
    same_pred = int(sum_comb(sum_cols))
    b = same_true - a  # Same True, Diff Pred
    c = same_pred - a  # Diff True, Same Pred
    d = nd - a - b - c  # Diff True, Diff Pred
    t = a
    s_plus = a + d
    s_minus = b + c

    return nd, s_plus, s_minus, t


def calculate_tau_score(y_true=None, y_pred=None, force_finite=True, finite_value=1.0):
    nd, s_plus, s_minus, t = compute_nd_splus_sminus_t(y_true, y_pred)
    den_sq = (nd - t) * nd
    if den_sq <= 0:
        return finite_value if force_finite else _raise_err("Tau", "partitions are identical or dataset is too small")
    return (s_plus - s_minus) / np.sqrt(den_sq)


def calculate_gamma_score(y_true=None, y_pred=None, force_finite=True, finite_value=1.0):
    if len(y_true) < 2:
        return finite_value if force_finite else _raise_err("Gamma", "dataset has fewer than 2 samples")
    return 2.0 * calculate_rand_score(y_true, y_pred) - 1.0


def calculate_gplus_score(y_true=None, y_pred=None, force_finite=True, finite_value=0.0):
    if len(y_true) < 2:
        return finite_value if force_finite else _raise_err("G-Plus", "dataset has fewer than 2 samples")
    return 1.0 - calculate_rand_score(y_true, y_pred)
