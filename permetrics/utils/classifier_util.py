#!/usr/bin/env python
# Created by "Thieu" at 12:38, 19/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


def calculate_confusion_matrix(y_true=None, y_pred=None, labels=None, normalize=None):
    """
    Compute the confusion matrix for classification tasks.

    The confusion matrix summarizes the performance of a classification model by comparing
    the predicted labels with the true labels. It can also normalize the matrix based on
    the specified normalization method.

    Args:
        y_true (array-like): Ground truth (correct) labels.
        y_pred (array-like): Predicted labels.
        labels (list, optional): Subset of labels to include in the matrix. Default is None.
        normalize (str, optional): Normalization method. One of {"true", "pred", "all"}.
            - "true": Normalize rows (true labels).
            - "pred": Normalize columns (predicted labels).
            - "all": Normalize the entire matrix.
            Default is None (no normalization).

    Returns:
        tuple:
            - matrix (ndarray): Confusion matrix (normalized if specified).
            - imap (dict): Mapping of labels to matrix indices.
            - imap_count (dict): Count of true labels for each class.

    Raises:
        ValueError: If specified labels do not exist in `y_true` or `y_pred`.
    """
    if y_true is None or y_pred is None:
        raise ValueError("y_true and y_pred must not be None.")

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.size == 0 or y_pred.size == 0 or y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must be non-empty and of equal length.")

    unique_all = sorted(np.unique(np.concatenate((y_true, y_pred))).tolist())
    unique_t, counts_t = np.unique(y_true, return_counts=True)
    imap_count_all = dict(zip(unique_t, counts_t))

    if labels is None:
        labels_selected = unique_all
    else:
        labels_selected = list(labels)
        if len(labels_selected) == 0:
            raise ValueError("Parameter 'labels' cannot be an empty list.")

    n_classes = len(labels_selected)
    imap = {key: i for i, key in enumerate(labels_selected)}
    imap_count = {lbl: int(imap_count_all.get(lbl, 0)) for lbl in labels_selected}

    matrix = np.zeros((n_classes, n_classes), dtype=int)
    # Only index the pairs (y_t, y_p) where BOTH belong to the labels_selected set.
    valid_mask = np.isin(y_true, labels_selected) & np.isin(y_pred, labels_selected)
    if np.any(valid_mask):
        y_t_idx = np.array([imap[y] for y in y_true[valid_mask]])
        y_p_idx = np.array([imap[y] for y in y_pred[valid_mask]])
        np.add.at(matrix, (y_t_idx, y_p_idx), 1)

    if normalize is None:
        return matrix, imap, imap_count

    if normalize not in {"true", "pred", "all"}:
        raise ValueError("normalize must be one of {'true', 'pred', 'all', None}")

    matrix_norm = matrix.astype(float)
    with np.errstate(all="ignore"):
        if normalize == "true":
            # [MATHEMATICAL TRUTH]: Divide by the actual original denominator of that class, (here is the different between Scikit-Learn and PerMetrics)
            real_supports = np.array([imap_count[lbl] for lbl in labels_selected], dtype=float)[:, None]
            matrix_norm /= real_supports
        elif normalize == "pred":
            matrix_norm /= matrix_norm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            matrix_norm /= float(y_true.size)
        matrix_norm = np.nan_to_num(matrix_norm)

    return matrix_norm, imap, imap_count


def calculate_single_label_metric(matrix, imap, imap_count, beta=1.0):
    """
    Compute various classification metrics for single-label classification.

    This function calculates metrics such as precision, recall, specificity, F1 score,
    Matthews correlation coefficient (MCC), and others for each class in the confusion matrix.

    Args:
        matrix (ndarray): Confusion matrix.
        imap (dict): Mapping of labels to matrix indices.
        imap_count (dict): Count of true labels for each class.
        beta (float, optional): Weight of recall in the F-beta score. Default is 1.0.

    Returns:
        dict: A dictionary where keys are class labels and values are dictionaries of metrics.
    """
    metrics = {}
    total_sum = float(matrix.sum())

    for label, idx in imap.items():
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - tp)
        fn = float(matrix[idx, :].sum() - tp)
        tn = total_sum - tp - fp - fn
        n_true = imap_count[label]

        prec = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) != 0 else 0.0

        p_o = (tp + tn) / total_sum if total_sum != 0 else 0.0
        p_e = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (total_sum ** 2) if total_sum != 0 else 0.0
        kappa = (p_o - p_e) / (1.0 - p_e) if (1.0 - p_e) != 0 else 0.0

        mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom != 0 else 0.0

        ls_denom = (tp + fn) / total_sum if total_sum != 0 else 0.0
        ls = prec / ls_denom if ls_denom != 0 else 0.0

        metrics[label] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_true": n_true,
            "precision": prec, "recall": rec, "specificity": spec,
            "negative_predictive_value": npv,
            "accuracy": p_o,
            "f1": (2 * prec * rec) / (prec + rec) if (prec + rec) != 0 else 0.0,
            "f2": (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) != 0 else 0.0,
            "fbeta": ((1 + beta ** 2) * prec * rec) / (beta ** 2 * prec + rec) if (beta ** 2 * prec + rec) != 0 else 0.0,
            "mcc": mcc,
            "hamming_loss": (fp + fn) / total_sum if total_sum != 0 else 0.0,
            "lift_score": ls,
            "jaccard_score": tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0.0,
            "kappa_score": kappa,
            "g_mean": np.sqrt(rec * spec)
        }
    return metrics


def calculate_accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):
    """
    Compute the accuracy score for classification tasks.

    Accuracy is the ratio of correctly predicted samples to the total number of samples.
    It can also compute weighted accuracy if sample weights are provided.

    Args:
        y_true (array-like): Ground truth (correct) labels.
        y_pred (array-like): Predicted labels.
        normalize (bool, optional): If True, return the fraction of correctly predicted samples.
            If False, return the number of correctly predicted samples. Default is True.
        sample_weight (array-like, optional): Sample weights. Default is None.

    Returns:
        float or int: Accuracy score (normalized or raw count).
    """
    matches = (y_true == y_pred)
    if sample_weight is not None:
        weights = np.asarray(sample_weight)
        weighted_matches = np.dot(matches, weights)
        return float(weighted_matches / np.sum(weights)) if normalize else float(weighted_matches)
    return float(np.mean(matches)) if normalize else int(np.sum(matches))


def calculate_class_support(y_true):
    """
    Compute the support (number of occurrences) for each class in the ground truth labels.

    Args:
        y_true (array-like): Ground truth (correct) labels.

    Returns:
        ndarray: Array of class counts.
    """
    _, counts = np.unique(y_true, return_counts=True)
    return counts


def calculate_roc_curve(y_true, y_score):
    """
    Compute the Receiver Operating Characteristic (ROC) curve.

    The ROC curve is a graphical representation of the trade-off between the true positive rate
    (sensitivity) and the false positive rate (1-specificity) at various threshold settings.

    Args:
        y_true (array-like): Ground truth (correct) binary labels.
        y_score (array-like): Predicted scores or probabilities for the positive class.

    Returns:
        tuple:
            - tpr (ndarray): True positive rates.
            - fpr (ndarray): False positive rates.
            - thresholds (ndarray): Thresholds used to compute TPR and FPR.

    Notes:
        - This function assumes `y_true` contains binary labels (0 and 1).
        - If only one class is present in `y_true`, the ROC curve is not defined.
    """
    desc_idx = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_idx]
    y_true = y_true[desc_idx]

    distinct_idx = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_idx, len(y_true) - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[y_score[0] + 1, y_score[threshold_idxs]]

    if tps[-1] == 0 or fps[-1] == 0:
        return np.zeros_like(thresholds), np.zeros_like(thresholds), thresholds

    return tps / tps[-1], fps / fps[-1], thresholds
