#!/usr/bin/env python
# Created by "Matt Q." at 23:05, 27/10/2022 --------%
#       Github: https://github.com/N3uralN3twork    %
#                                                   %
# Improved by: "Thieu" at 17:10, 25/07/2023 --------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import entropy as calculate_entropy
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
    Args:
        X: The X matrix features
        y_pred: The predicted results
        force_finite: Make result as finite number
        finite_value: The value that used to replace the infinite value or NaN value.

    Returns:
        The Calinski Harabasz Index
    """
    n_samples, _ = X.shape
    n_clusters = len(np.unique(y_pred))
    if n_clusters == 1:
        if force_finite:
            return finite_value
        else:
            raise ValueError("The Calinski-Harabasz index is undefined when y_pred has only 1 cluster.")
    overall_mean = np.mean(X, axis=0)
    # Calculate between-cluster variance and cluster sizes
    cluster_sizes = np.bincount(y_pred, minlength=n_clusters)
    cluster_means = np.array([np.mean(X[y_pred == i], axis=0) for i in range(n_clusters)])
    between_var = np.sum(cluster_sizes * np.sum((cluster_means - overall_mean) ** 2, axis=1))
    # Calculate within-cluster variance
    within_var = np.sum((X - cluster_means[y_pred]) ** 2)
    # Calculate the CH Index
    res = (between_var / within_var) * ((n_samples - n_clusters) / (n_clusters - 1))
    return res


def calculate_xie_beni_index(X=None, y_pred=None, force_finite=True, finite_value=1e10):
    n_clusters = len(np.unique(y_pred))
    if n_clusters == 1:
        if force_finite:
            return finite_value
        else:
            raise ValueError("The Xie-Beni index is undefined when y_pred has only 1 cluster.")
    # Get the centroids
    centroids, _ = compute_barycenters(X, y_pred)
    wgss = np.sum(np.min(cdist(X, centroids, metric='euclidean'), axis=1) ** 2)
    # Computing the minimum squared distance to the centroids:
    MinSqDist = np.min(pdist(centroids, metric='sqeuclidean'))
    res = (wgss / X.shape[0]) / MinSqDist
    return res


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


def calculate_dunn_index(X=None, y_pred=None, use_modified=True, force_finite=True, finite_value=0.):
    centers, _ = compute_barycenters(X, y_pred)
    n_clusters = len(centers)
    if n_clusters == 1:
        if force_finite:
            return finite_value
        else:
            raise ValueError("The Dunn index is undefined when y_pred has only 1 cluster.")
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
    return dmin / dmax


def calculate_ksq_detw_index(X=None, y_pred=None, use_normalized=True):
    centers, _ = compute_barycenters(X, y_pred)
    scatter_matrices = np.zeros((X.shape[1], X.shape[1]))  # shape of (n_features, n_features)
    for kdx in range(len(centers)):
        X_k = X[y_pred == kdx]
        scatter_matrices += compute_WG(X_k)
    if use_normalized:
        scatter_matrices = (scatter_matrices - np.min(scatter_matrices)) / (np.max(scatter_matrices) - np.min(scatter_matrices))
    res = len(centers) ** 2 * np.linalg.det(scatter_matrices)
    return res


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
    pred_labels = np.unique(y_pred)
    n_clusters = len(pred_labels)
    if n_clusters == 1:
        if force_finite:
            return finite_value
        else:
            raise ValueError("The Beale index is undefined when y_pred has only 1 cluster.")
    n_samples, n_features = X.shape
    centers, _ = compute_barycenters(X, y_pred)
    sse_within = 0
    sse_between = 0
    for k in pred_labels:
        sse_within += np.sum((X[y_pred == k] - centers[k]) ** 2)
        sse_between += np.sum((centers[k] - np.mean(X, axis=0)) ** 2)
    df_within = n_samples - n_clusters
    df_between = n_clusters - 1
    ms_within = sse_within / df_within
    ms_between = sse_between / df_between
    result = ms_within / ms_between
    return result


def calculate_r_squared_index(X=None, y_pred=None):
    tss = compute_TSS(X)
    return (tss - compute_WGSS(X, y_pred)) / tss if tss > 0 else 1.0


def calculate_density_based_clustering_validation_index(X=None, y_pred=None, force_finite=True, finite_value=1.):
    pred_labels = np.unique(y_pred)
    n_clusters = len(pred_labels)
    if n_clusters == 1:
        if force_finite:
            return finite_value
        else:
            raise ValueError("The Density-based Clustering Validation Index is undefined when y_pred has only 1 cluster.")
    n_samples, n_features = X.shape
    centroids = np.zeros((n_clusters, n_features))
    for k in pred_labels:
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
    return result


def calculate_hartigan_index(X=None, y_pred=None, force_finite=True, finite_value=1e10):
    centroids, _ = compute_barycenters(X, y_pred)
    num_clusters = len(np.unique(y_pred))
    if num_clusters == 1:
        if force_finite:
            return finite_value
        else:
            raise ValueError("The Hartigan Index is undefined when y_pred has only 1 cluster.")
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
    return hi


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
    if h + c == 0:
        res = 0
    else:
        res = ((1+beta) * h * c) / (beta*h + c)
    return res


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


def calculate_russel_rao_score(y_true=None, y_pred=None):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    NT = yy + yn + ny + nn
    return yy / NT if NT > 0 else 0.0


def calculate_sokal_sneath1_score(y_true=None, y_pred=None):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    den = yy + 2 * (yn + ny)
    return yy / den if den > 0 else 0.0


def calculate_sokal_sneath2_score(y_true=None, y_pred=None):
    yy, yn, ny, nn = compute_confusion_matrix(y_true, y_pred, normalize=True)
    den = yy + nn + 0.5 * (yn + ny)
    return (yy + nn) / den if den > 0 else 0.0


# def calculate_purity_score(y_true=None, y_pred=None):
#     # Find the number of data points
#     N = len(y_true)
#     # Find the unique class labels in the true labels
#     unique_classes = np.unique(y_true)
#     # Initialize the purity score
#     purity = 0
#     # Iterate over each unique class label
#     for c in unique_classes:
#         # Find the indices of data points with the current class label in the true labels
#         class_indices = np.where(y_true == c)[0]
#         # Find the corresponding predicted labels for these data points
#         class_predictions = y_pred[class_indices]
#         # Count the occurrences of each predicted label
#         class_predictions = np.round(class_predictions).astype(int)
#         class_counts = np.bincount(class_predictions)
#         # Add the size of the majority class to the purity score
#         purity += np.max(class_counts)
#     # Normalize the purity score by dividing by the total number of data points
#     return purity / N


def calculate_purity_score(y_true=None, y_pred=None):
    """
    O(N) Vectorized Purity Score. Safe for arbitrary label formats.
    """
    contingency = compute_contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency, axis=0)) / len(y_true)


# def calculate_entropy_score(y_true=None, y_pred=None):
#     # Find the number of data points
#     N = len(y_true)
#     # Find the unique class labels in the true labels
#     unique_classes = np.unique(y_true)
#     result = 0
#     # Iterate over each unique class label
#     for c in unique_classes:
#         # Find the indices of data points with the current class label in the true labels
#         class_indices = np.where(y_true == c)[0]
#         # Find the corresponding predicted labels for these data points
#         class_predictions = y_pred[class_indices]
#         class_predictions = np.round(class_predictions).astype(int)
#         # Count the occurrences of each predicted label
#         class_counts = np.bincount(class_predictions)
#         # Normalize the class counts by dividing by the total number of data points in the cluster
#         class_distribution = class_counts / len(class_predictions)
#         # Compute the entropy of the cluster
#         cluster_entropy = calculate_entropy(class_distribution, base=2)
#         # Weight the entropy by the relative size of the cluster
#         cluster_size = len(class_indices)
#         result += (cluster_size / N) * cluster_entropy
#     return result


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
    nd, s_plus, s_minus, t = compute_nd_splus_sminus_t(y_true, y_pred)
    den = s_plus + s_minus
    if den == 0:
        return finite_value if force_finite else _raise_err("Gamma", "no sample pairs available for comparison")
    return (s_plus - s_minus) / den


def calculate_gplus_score(y_true=None, y_pred=None, force_finite=True, finite_value=0.0):
    nd, s_plus, s_minus, t = compute_nd_splus_sminus_t(y_true, y_pred)
    if nd == 0:
        return finite_value if force_finite else _raise_err("G-Plus", "dataset has fewer than 2 samples")
    return s_minus / nd
