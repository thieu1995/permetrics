# !/usr/bin/env python
# Created by "Matt Q." at 23:05, 27/10/2022 --------%
#       Github: https://github.com/N3uralN3twork    %
#                                                   %
# Improved by: "Thieu" at 17:10, 25/07/2023 --------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import time

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial import distance_matrix
from scipy.stats import entropy as calculate_entropy


def compute_clusters(labels):
    """
    Get the dict of clusters and dict of cluster size
    """
    dict_clusters = {}
    for idx, label in enumerate(labels):
        if label in dict_clusters:
            dict_clusters[label].append(idx)
        else:
            dict_clusters[label] = [idx]
    dict_cluster_sizes = {}
    for label, group in dict_clusters.items():
        dict_cluster_sizes[label] = len(group)
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


def get_min_dist(X, centers):
    """
    Get the min distance from samples X to centers
    """
    dist = cdist(X, centers, metric='euclidean')
    min_dist = np.min(dist, axis=1)
    return min_dist


def get_centroids(X, labels):
    """
    Calculates the centroids from the data given, for each class.

    Args:
        X (pd.DataFrame, np.ndarray): The original data that was clustered
        labels (list, np.ndarray): The predicted cluster assignment values

    Returns:
        centroids (np.ndarray): The centroids given the input data and labels
    """
    n_samples, n_features = X.shape
    n_classes = len(np.unique(labels))
    # * Mask mapping each class to its members.
    centroids = np.empty((n_classes, n_features), dtype=np.float64)
    # * Number of clusters in each class.
    nk = np.zeros(n_classes)
    for k in range(n_classes):
        centroid_mask = labels == k
        nk[k] = np.sum(centroid_mask)
        centroids[k] = X[centroid_mask].mean(axis=0)
    return centroids


def compute_contingency_matrix(y_true, y_pred):
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    num_true = len(unique_true)
    num_pred = len(unique_pred)
    contingency = np.zeros((num_true, num_pred), dtype=np.int64)
    for i in range(len(y_true)):
        true_label = np.where(unique_true == y_true[i])[0]
        pred_label = np.where(unique_pred == y_pred[i])[0]
        contingency[true_label, pred_label] += 1
    return contingency


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


def compute_homogeneity(y_true, y_pred):
    h_labels_true = compute_entropy(y_true)
    h_labels_true_given_pred = compute_conditional_entropy(y_true, y_pred)
    if h_labels_true == 0:
        return 1
    else:
        return 1 - h_labels_true_given_pred / h_labels_true


def compute_confusion_matrix(y_true, y_pred, normalize=False):
    """
    Computes the confusion matrix for a clustering problem given the true labels and the predicted labels.
    http://cran.nexr.com/web/packages/clusterCrit/vignettes/clusterCrit.pdf
    """
    n = len(y_true)
    yy, yn, ny, nn = 0, 0, 0, 0
    for i in range(n):
        for j in range(i+1, n):
            if y_true[i] == y_true[j] and y_pred[i] == y_pred[j]:
                yy += 1
            elif y_true[i] == y_true[j] and y_pred[i] != y_pred[j]:
                yn += 1
            elif y_true[i] != y_true[j] and y_pred[i] == y_pred[j]:
                ny += 1
            else:
                nn += 1
    res = np.array([yy, yn, ny, nn])
    if normalize:
        return res/np.sum(res)
    return res


def calculate_sum_squared_error_index(X=None, y_pred=None, decimal=6):
    centers, _ = compute_barycenters(X, y_pred)
    centroid_distances = centers[y_pred]
    squared_distances = np.sum((X - centroid_distances) ** 2, axis=1)
    return np.round(np.sum(squared_distances), decimal)


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
