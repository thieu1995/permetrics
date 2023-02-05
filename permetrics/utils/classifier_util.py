#!/usr/bin/env python
# Created by "Thieu" at 12:38, 19/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


def calculate_confusion_matrix(y_true=None, y_pred=None, labels=None, normalize=None):
    """
    Generate a confusion matrix for multiple classification

    Args:
        y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
        y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
        labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
        normalize ('true', 'pred', 'all', None): Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.

    Returns:
        matrix (np.ndarray): a 2-dimensional list of pairwise counts
        imap (dict): a map between label and index of confusion matrix
        imap_count (dict): a map between label and number of true label in y_true
    """

    # Get values by label
    unique, counts = np.unique(y_true, return_counts=True)
    imap_count = {unique[idx]: counts[idx] for idx in range(len(unique))}
    unique = sorted(unique)
    matrix = [[0 for _ in unique] for _ in unique]
    imap = {key: i for i, key in enumerate(unique)}

    # Generate Confusion Matrix
    for p, a in zip(y_true, y_pred):
        matrix[imap[p]][imap[a]] += 1

    # Matrix Normalization
    matrix = np.array(matrix)
    with np.errstate(all="ignore"):
        if normalize == "true":
            matrix_normalized = matrix / matrix.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            matrix_normalized = matrix / matrix.sum(axis=0, keepdims=True)
        elif normalize == "all":
            matrix_normalized = matrix / matrix.sum()
        else:
            matrix_normalized = matrix
        matrix_normalized = np.nan_to_num(matrix_normalized)

    # Matrix Normalization
    # matrix = np.array(matrix)
    # matrix_normalized = np.zeros_like(matrix).astype(float)
    # if normalize == "true":
    #     sigma = np.sum(matrix, axis=1)
    #     for idx in range(len(unique)):
    #         matrix_normalized[idx, :] = 0.0 if sigma[idx] == 0 else matrix[idx, :] / sigma[idx]
    # elif normalize == "pred":
    #     sigma = np.sum(matrix, axis=0)
    #     for idx in range(len(unique)):
    #         matrix_normalized[:, idx] = 0.0 if sigma[idx] == 0 else matrix[:, idx] / sigma[idx]
    # elif normalize == "all":
    #     matrix_normalized = matrix / np.sum(matrix)
    # else:
    #     matrix_normalized = matrix

    # Get values by label
    if labels is None:
        return matrix_normalized, imap, imap_count
    elif isinstance(labels, (list, tuple, np.ndarray)):
        labels = list(labels)
        if np.all(np.isin(labels, unique)):
            matrix_final = [[0 for _ in labels] for _ in labels]
            imap_final = {key: i for i, key in enumerate(labels)}
            imap_count_final = {label: imap_count[label] for label in labels}
            for label1 in labels:
                for label2 in labels:
                    matrix_final[imap_final[label1]][imap_final[label2]] = matrix_normalized[imap[label1]][imap[label2]]
            return np.array(matrix_final), imap_final, imap_count_final
        else:
            raise TypeError("All specified label should be in y_true!")
    else:
        raise TypeError("Labels should be a tuple / a list / a numpy array!")


def calculate_single_label_metric(matrix, imap, imap_count, beta=1.0):
    """
    Generate a dictionary of supported metrics for each label

    Args:
        matrix (np.ndarray): a 2-dimensional list of pairwise counts
        imap (dict): a map between label and index of confusion matrix
        imap_count (dict): a map between label and number of true label in y_true
        beta (float): to calculate the f-beta score

    Returns:
        dict_metrics (dict): a dictionary of supported metrics
    """
    metrics = {}
    with np.errstate(all="ignore"):
        for label, idx in imap.items():
            metric = {}
            tp = matrix[idx][idx]
            fp = matrix[:, idx].sum() - tp
            fn = matrix[idx, :].sum() - tp
            tn = matrix.sum() - tp - fp - fn
            n_true = imap_count[label]
            precision = tp / (tp+fp)
            recall = tp / (tp + fn)
            mcc = (tp * tn - fp * fn) / np.sqrt(((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
            ls = (tp/(tp + fp)) / ((tp + fn) / (tp + tn + fp + fn))
            jsi = tp / (tp + fp + fn)

            metric["tp"] = tp
            metric["fp"] = fp
            metric["fn"] = fn
            metric["tn"] = tn
            metric["n_true"] = n_true

            metric["precision"] = np.nan_to_num(precision, nan=0.0, posinf=0.0, neginf=0.0)
            metric["recall"] = np.nan_to_num(recall, nan=0.0, posinf=0.0, neginf=0.0)
            metric["specificity"] = np.nan_to_num(tn / (tn + fp), nan=0.0, posinf=0.0, neginf=0.0)
            metric["negative_predictive_value"] = np.nan_to_num(tn / (tn + fn), nan=0.0, posinf=0.0, neginf=0.0)
            metric["accuracy"] = np.nan_to_num((tp + tn) / (tp + tn + fp + fn), nan=0.0, posinf=0.0, neginf=0.0)
            metric["f1"] = np.nan_to_num((2 * recall * precision) / (recall + precision), nan=0.0, posinf=0.0, neginf=0.0)
            metric["f2"] = np.nan_to_num((5 * precision * recall) / (4 * precision + recall), nan=0.0, posinf=0.0, neginf=0.0)
            metric["fbeta"] = np.nan_to_num(((1+beta**2) * precision*recall) / (beta**2 * precision + recall), nan=0.0, posinf=0.0, neginf=0.0)
            metric["mcc"] = np.nan_to_num(mcc, nan=0.0, posinf=0.0, neginf=0.0)
            metric["hamming_loss"] = np.nan_to_num(1.0 - tp / matrix.sum(), nan=0.0, posinf=0.0, neginf=0.0)
            metric["lift_score"] = np.nan_to_num(ls, nan=0.0, posinf=0.0, neginf=0.0)
            metric["jaccard_similarities"] = np.nan_to_num(jsi, nan=0.0, posinf=0.0, neginf=0.0)
            metrics[label] = metric

        # list_precision = np.nan_to_num([value[0] / (value[0] + value[1]) for value in cm.values()])
        # list_recall = np.nan_to_num([value[0] / (value[0] + value[3]) for value in cm.values()])
        # list_weights = np.nan_to_num([value[4] for value in cm.values()])
    return metrics


class LabelEncoder:
    def __init__(self):
        self.classes_ = None
        self.encoded_classes_ = None

    def fit(self, y):
        self.classes_, indices = np.unique(y, return_inverse=True)
        self.encoded_classes_ = np.arange(len(self.classes_))
        return self

    def transform(self, y):
        return np.searchsorted(self.classes_, y)

    def inverse_transform(self, y):
        return self.classes_[y]

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)


# class LabelEncoder:
#     def __init__(self):
#         self.classes_ = None
#         self.encoded_classes_ = None
#
#     def fit(self, y):
#         self.classes_ = list(set(y))
#         self.encoded_classes_ = {c: i for i, c in enumerate(self.classes_)}
#         return self
#
#     def transform(self, y):
#         return [self.encoded_classes_[c] for c in y]
#
#     def inverse_transform(self, y):
#         return [self.classes_[i] for i in y]































