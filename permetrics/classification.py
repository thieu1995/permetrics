#!/usr/bin/env python
# Created by "Thieu" at 09:29, 23/09/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from permetrics.evaluator import Evaluator
from permetrics.utils import data_util as du
from permetrics.utils import classifier_util as cu
import numpy as np


class ClassificationMetric(Evaluator):
    """
    A class for evaluating classification metrics.

    Parameters
    ----------
    y_true : tuple, list, np.ndarray, optional
        The ground truth values. Default is None.

    y_pred : tuple, list, np.ndarray, optional
        The predicted values. Default is None.

    labels : tuple, list, np.ndarray, optional
        List of labels to index the matrix. This may be used to reorder or select a subset of labels. Default is None.

    pos_label : int or str
        Positive label for binary classification.

    average : str or None, optional
        Determines the type of averaging performed on the data. Options are:
        - 'binary': Calculate for binary classification problem
        - 'micro': Calculate metrics globally by considering each element of the label indicator matrix as a label.
        - 'macro': Calculate metrics for each label and find their unweighted mean.
        - 'weighted': Calculate metrics for each label and find their average, weighted by support.
        - None: Scores for each class are returned.
        Default is "binary".

    Methods
    -------
    get_support(name=None, verbose=True)
        Retrieve the support information for a specific metric or all metrics.

    get_processed_data(y_true=None, y_pred=None)
        Process and format the input data for evaluation.

    get_processed_data2(y_true=None, y_pred=None)
        Process and format the input data for ROC and probability-based metrics.

    precision_score(...)
        Calculate the precision score.

    negative_predictive_value(...)
        Calculate the negative predictive value.

    specificity_score(...)
        Calculate the specificity score.

    recall_score(...)
        Calculate the recall score.

    f1_score(...)
        Calculate the F1 score.

    f2_score(...)
        Calculate the F2 score.

    fbeta_score(...)
        Calculate the F-beta score.

    matthews_correlation_coefficient(...)
        Calculate the Matthews correlation coefficient.

    hamming_loss(...)
        Calculate the hamming loss.

    lift_score(...)
        Calculate the lift score.

    cohen_kappa_score(...)
        Calculate the Cohen's kappa score.

    jaccard_similarity_index(...)
        Calculate the Jaccard similarity index.

    g_mean_score(...)
        Calculate the geometric mean score.

    accuracy_score(...)
        Calculate the accuracy score.

    confusion_matrix(...)
        Generate the confusion matrix.

    roc_auc_score(...)
        Calculate the ROC-AUC score.

    gini_index(...)
        Calculate the Gini index.

    brier_score_loss(...)
        Calculate the Brier score loss.

    crossentropy_loss(...)
        Calculate the cross-entropy loss.

    hinge_loss(...)
        Calculate the hinge loss.

    kullback_leibler_divergence_loss(...)
        Calculate the Kullback-Leibler divergence loss.
    """

    SUPPORT = {
        "AS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "PS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "NPV": {"type": "max", "range": "[0, 1]", "best": "1"},
        "RS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "SS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "F1S": {"type": "max", "range": "[0, 1]", "best": "1"},
        "F2S": {"type": "max", "range": "[0, 1]", "best": "1"},
        "FBS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "MCC": {"type": "max", "range": "[-1, +1]", "best": "1"},
        "CKS": {"type": "max", "range": "[-1, +1]", "best": "1"},
        "JSI": {"type": "max", "range": "[0, 1]", "best": "1"},
        "JSS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "GMS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "ROC-AUC": {"type": "max", "range": "[0, 1]", "best": "1"},
        "ROC": {"type": "max", "range": "[0, 1]", "best": "1"},
        "AUC": {"type": "max", "range": "[0, 1]", "best": "1"},
        "GINI": {"type": "max", "range": "[-1, 1]", "best": "1"},
        "LS": {"type": "max", "range": "[0, +inf)", "best": "unknown"},

        "CEL": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "HML": {"type": "min", "range": "[0, 1]", "best": "0"},
        "HGL": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "KLDL": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "BSL": {"type": "min", "range": "[0, 1]", "best": "0"}
    }

    def __init__(self, y_true=None, y_pred=None, **kwargs):
        super().__init__(y_true, y_pred, **kwargs)
        if kwargs is None: kwargs = {}
        self.set_keyword_arguments(kwargs)
        self.binary = True
        self.representor = "number"     # "number" or "string"
        self.le = None  # LabelEncoder

    @staticmethod
    def get_support(name=None, verbose=True):
        """
        Retrieve the support information for a specific metric or all metrics.

        Parameters
        ----------
        name : str, optional
            Name of the metric to retrieve. Use "all" to retrieve all metrics.
        verbose : bool, optional
            Whether to print the metric details.

        Returns
        -------
        dict
            Support information for the specified metric(s).
        """
        if name == "all":
            if verbose:
                for key, value in ClassificationMetric.SUPPORT.items():
                    print(f"Metric {key} : {value}")
            return ClassificationMetric.SUPPORT
        if name not in list(ClassificationMetric.SUPPORT.keys()):
            raise ValueError(f"ClassificationMetric doesn't support metric named: {name}")
        if verbose:
            print(f"Metric {name}: {ClassificationMetric.SUPPORT[name]}")
        return ClassificationMetric.SUPPORT[name]

    def get_processed_data(self, y_true=None, y_pred=None):
        """
        Process and format the input data for evaluation.

        Returns:
            y_true_final: y_true used in evaluation process.
            y_pred_final: y_pred used in evaluation process
            unique_classes: All unique classes from y_true and y_pred
            representor: the label is number or string
        """
        if (y_true is not None) and (y_pred is not None):
            return du.format_classification_data(y_true, y_pred)
        if (self.y_true is not None) and (self.y_pred is not None):
            return du.format_classification_data(self.y_true, self.y_pred)
        raise ValueError("y_true or y_pred is None. You need to pass y_true and y_pred to object creation or function called.")

    def get_processed_data2(self, y_true=None, y_pred=None):
        """
        Returns:
            y_true_final: y_true used in evaluation process.
            y_pred_final: y_pred used in evaluation process
            binary: is problem binary or multi-class classification
            representor: the label is number or string
        """
        if (y_true is not None) and (y_pred is not None):
            return du.format_y_score(y_true, y_pred)
        if (self.y_true is not None) and (self.y_pred is not None):
            return du.format_y_score(self.y_true, self.y_pred)
        raise ValueError("y_true or y_pred is None. You need to pass y_true and y_pred to object creation or function called.")

    def _get_micro_stats(self, matrix):
        """Helper calculates accurate global components for multi-class classification"""
        N = matrix.sum()
        K = matrix.shape[0]
        tp = np.trace(matrix)
        fp = fn = N - tp
        tn = N * K - (tp + fp + fn)
        return tp, fp, fn, tn

    def _aggregate(self, metric_key, y_true, y_pred, labels, pos_label, average, beta=1.0):
        """
        Aggregate metrics based on the specified averaging method.

        Parameters
        ----------
        metric_key : str
            Metric key to calculate.
        y_true : array-like
            Ground truth values.
        y_pred : array-like
            Predicted values.
        labels : list, optional
            List of labels to consider.
        pos_label : int or str
            Positive label for binary classification.
        average : str or None
            Averaging method ('binary', 'micro', 'macro', 'weighted', or None).
        beta : float, optional
            Weight of recall in the F-beta score.

        Returns
        -------
        float or dict
            Aggregated metric value(s).
        """
        y_true, y_pred, unique_classes, _ = self.get_processed_data(y_true, y_pred)

        # 1. Check binary classification problem
        if average == "binary":
            if len(unique_classes) > 2:
                raise ValueError(f"Target is multiclass ({len(unique_classes)} classes) but average='binary'. "
                    "Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].")
            if len(unique_classes) > 1 and pos_label not in unique_classes:
                raise ValueError(f"pos_label={pos_label} is not a valid label. Unique labels are {unique_classes}")

        # 2. Calculate the original Confusion Matrix
        matrix, imap, imap_count = cu.calculate_confusion_matrix(y_true, y_pred, labels=None, normalize=None)

        # 3. Micro: Combining multi-class problems into original composite 2x2 matrices
        if average == "micro":
            tp, fp, fn, tn = self._get_micro_stats(matrix)
            m_micro = np.array([[tp, fn], [fp, tn]], dtype=float)
            res_micro = cu.calculate_single_label_metric(m_micro, {"_m": 0}, {"_m": tp + fn}, beta=beta)
            return float(res_micro["_m"][metric_key])

        # 4. Calculate all classes
        all_metrics = cu.calculate_single_label_metric(matrix, imap, imap_count, beta=beta)

        # 5. Binary (Returns the correct float of pos_label)
        if average == "binary":
            return float(all_metrics[pos_label][metric_key]) if pos_label in all_metrics else 0.0

        target_labels = list(labels) if labels is not None else list(imap.keys())
        if not np.all(np.isin(target_labels, unique_classes)):
            raise ValueError("Specified labels do not exist in data.")

        # 6. None (Returns a dict based on the label)
        if average is None:
            return {lbl: all_metrics[lbl][metric_key] for lbl in target_labels if lbl in all_metrics}

        vals = np.array([all_metrics[lbl][metric_key] for lbl in target_labels if lbl in all_metrics], dtype=float)
        supps = np.array([all_metrics[lbl]["n_true"] for lbl in target_labels if lbl in all_metrics], dtype=float)

        if average == "macro":
            return float(np.mean(vals)) if len(vals) > 0 else 0.0
        if average == "weighted":
            total_s = np.sum(supps)
            return float(np.dot(vals, supps) / total_s) if total_s > 0 else 0.0

        raise ValueError(f"Unsupported average setting: {average}")

    def precision_score(self, y_true=None, y_pred=None, labels=None, pos_label=1, average="binary", **kwargs):
        """
        Parameters
        ----------
        y_true : array-like, optional
            Ground truth values.
        y_pred : array-like, optional
            Predicted values.
        labels : list, optional
            List of labels to include in the calculation.
        pos_label : int or str, optional
            The positive class label for binary classification.
        average : str, optional
            Averaging method ('binary', 'micro', 'macro', 'weighted', or None).

        Returns
        -------
        float or dict
            Precision score.
        """
        return self._aggregate("precision", y_true, y_pred, labels, pos_label, average)

    def negative_predictive_value(self, y_true=None, y_pred=None, labels=None, pos_label=1, average="binary", **kwargs):
        """
        Calculate the negative predictive value.

        Parameters
        ----------
        y_true : array-like, optional
            Ground truth values.
        y_pred : array-like, optional
            Predicted values.
        labels : list, optional
            List of labels to include in the calculation.
        pos_label : int or str, optional
            The positive class label for binary classification.
        average : str, optional
            Averaging method ('binary', 'micro', 'macro', 'weighted', or None).

        Returns
        -------
        float or dict
            Negative predictive value.
        """
        return self._aggregate("negative_predictive_value", y_true, y_pred, labels, pos_label, average)

    def specificity_score(self, y_true=None, y_pred=None, labels=None, pos_label=1, average="binary", **kwargs):
        """
        Calculate the specificity score.

        Parameters
        ----------
        y_true : array-like, optional
            Ground truth values.
        y_pred : array-like, optional
            Predicted values.
        labels : list, optional
            List of labels to include in the calculation.
        pos_label : int or str, optional
            The positive class label for binary classification.
        average : str, optional
            Averaging method ('binary', 'micro', 'macro', 'weighted', or None).

        Returns
        -------
        float or dict
            Specificity score.
        """
        return self._aggregate("specificity", y_true, y_pred, labels, pos_label, average)

    def recall_score(self, y_true=None, y_pred=None, labels=None, pos_label=1, average="binary", **kwargs):
        """
        Parameters
        ----------
        y_true : array-like, optional
            Ground truth values.
        y_pred : array-like, optional
            Predicted values.
        labels : list, optional
            List of labels to include in the calculation.
        pos_label : int or str, optional
            The positive class label for binary classification.
        average : str, optional
            Averaging method ('binary', 'micro', 'macro', 'weighted', or None).

        Returns
        -------
        float or dict
            Recall score.
        """
        return self._aggregate("recall", y_true, y_pred, labels, pos_label, average)

    def f1_score(self, y_true=None, y_pred=None, labels=None, pos_label=1, average="binary", **kwargs):
        """
        Parameters
        ----------
        y_true : array-like, optional
            Ground truth values.
        y_pred : array-like, optional
            Predicted values.
        labels : list, optional
            List of labels to include in the calculation.
        pos_label : int or str, optional
            The positive class label for binary classification.
        average : str, optional
            Averaging method ('binary', 'micro', 'macro', 'weighted', or None).

        Returns
        -------
        float or dict
            F1 score.
        """
        return self._aggregate("f1", y_true, y_pred, labels, pos_label, average)

    def f2_score(self, y_true=None, y_pred=None, labels=None, pos_label=1, average="binary", **kwargs):
        """
        Parameters
        ----------
        y_true : array-like, optional
            Ground truth values.
        y_pred : array-like, optional
            Predicted values.
        labels : list, optional
            List of labels to include in the calculation.
        pos_label : int or str, optional
            The positive class label for binary classification.
        average : str, optional
            Averaging method ('binary', 'micro', 'macro', 'weighted', or None).

        Returns
        -------
        float or dict
            F2 score.
        """
        return self._aggregate("f2", y_true, y_pred, labels, pos_label, average)

    def fbeta_score(self, y_true=None, y_pred=None, beta=1.0, labels=None, pos_label=1, average="binary", **kwargs):
        """
        Parameters
        ----------
        y_true : array-like, optional
            Ground truth values.
        y_pred : array-like, optional
            Predicted values.
        beta : float, optional
            Weight of recall in the F-beta score.
        labels : list, optional
            List of labels to include in the calculation.
        pos_label : int or str, optional
            The positive class label for binary classification.
        average : str, optional
            Averaging method ('binary', 'micro', 'macro', 'weighted', or None).

        Returns
        -------
        float or dict
            F-beta score.
        """
        return self._aggregate("fbeta", y_true, y_pred, labels, pos_label, average, beta=beta)

    def matthews_correlation_coefficient(self, y_true=None, y_pred=None, labels=None, pos_label=1, average="binary", **kwargs):
        """
        Parameters
        ----------
        y_true : array-like, optional
            Ground truth values.
        y_pred : array-like, optional
            Predicted values.
        labels : list, optional
            List of labels to include in the calculation.
        pos_label : int or str, optional
            The positive class label for binary classification.
        average : str, optional
            Averaging method ('binary', 'micro', 'macro', 'weighted', or None).

        Returns
        -------
        float or dict
            Matthews correlation coefficient.
        """
        return self._aggregate("mcc", y_true, y_pred, labels, pos_label, average)

    def hamming_loss(self, y_true=None, y_pred=None, labels=None, pos_label=1, average="binary", **kwargs):
        """
        Parameters
        ----------
        y_true : array-like, optional
            Ground truth values.
        y_pred : array-like, optional
            Predicted values.
        labels : list, optional
            List of labels to include in the calculation.
        pos_label : int or str, optional
            The positive class label for binary classification.
        average : str, optional
            Averaging method ('binary', 'micro', 'macro', 'weighted', or None).

        Returns
        -------
        float or dict
            Hamming loss.
        """
        return self._aggregate("hamming_loss", y_true, y_pred, labels, pos_label, average)

    def lift_score(self, y_true=None, y_pred=None, labels=None, pos_label=1, average="binary", **kwargs):
        """
        Parameters
        ----------
        y_true : array-like, optional
            Ground truth values.
        y_pred : array-like, optional
            Predicted values.
        labels : list, optional
            List of labels to include in the calculation.
        pos_label : int or str, optional
            The positive class label for binary classification.
        average : str, optional
            Averaging method ('binary', 'micro', 'macro', 'weighted', or None).

        Returns
        -------
        float or dict
            Lift score.
        """
        return self._aggregate("lift_score", y_true, y_pred, labels, pos_label, average)

    def cohen_kappa_score(self, y_true=None, y_pred=None, labels=None, pos_label=1, average="binary", **kwargs):
        """
        Parameters
        ----------
        y_true : array-like, optional
            Ground truth values.
        y_pred : array-like, optional
            Predicted values.
        labels : list, optional
            List of labels to include in the calculation.
        pos_label : int or str, optional
            The positive class label for binary classification.
        average : str, optional
            Averaging method ('binary', 'micro', 'macro', 'weighted', or None).

        Returns
        -------
        float or dict
            Cohen's kappa score.
        """
        return self._aggregate("kappa_score", y_true, y_pred, labels, pos_label, average)

    def jaccard_similarity_index(self, y_true=None, y_pred=None, labels=None, pos_label=1, average="binary", **kwargs):
        """
        Parameters
        ----------
        y_true : array-like, optional
            Ground truth values.
        y_pred : array-like, optional
            Predicted values.
        labels : list, optional
            List of labels to include in the calculation.
        pos_label : int or str, optional
            The positive class label for binary classification.
        average : str, optional
            Averaging method ('binary', 'micro', 'macro', 'weighted', or None).

        Returns
        -------
        float or dict
            Jaccard similarity index.
        """
        return self._aggregate("jaccard_score", y_true, y_pred, labels, pos_label, average)

    def g_mean_score(self, y_true=None, y_pred=None, labels=None, pos_label=1, average="binary", **kwargs):
        """
        Parameters
        ----------
        y_true : array-like, optional
            Ground truth values.
        y_pred : array-like, optional
            Predicted values.
        labels : list, optional
            List of labels to include in the calculation.
        pos_label : int or str, optional
            The positive class label for binary classification.
        average : str, optional
            Averaging method ('binary', 'micro', 'macro', 'weighted', or None).

        Returns
        -------
        float or dict
            Geometric mean (G-mean) score.
        """
        return self._aggregate("g_mean", y_true, y_pred, labels, pos_label, average)

    def accuracy_score(self, y_true=None, y_pred=None, normalize=True, sample_weight=None, **kwargs):
        """
        Parameters
        ----------
        y_true : array-like, optional
            Ground truth (correct) target values.
        y_pred : array-like, optional
            Estimated target values.
        normalize : bool, optional
            If True, return the fraction of correctly classified samples (float).
            If False, return the number of correctly classified samples (int).
        sample_weight : array-like, optional
            Sample weights.

        Returns
        -------
        float or int
            Accuracy score.
        """
        y_true, y_pred, _, _ = self.get_processed_data(y_true, y_pred)
        return cu.calculate_accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight)

    # =====================================================================
    # ROC & LOSS FUNCTIONS
    # =====================================================================

    def confusion_matrix(self, y_true=None, y_pred=None, labels=None, normalize=None, **kwargs):
        """
        Parameters
        ----------
        y_true : array-like, optional
            Ground truth (correct) target values.
        y_pred : array-like, optional
            Estimated target values.
        labels : list, optional
            List of labels to index the matrix.
        normalize : str or None, optional
            Normalization mode ('true', 'pred', 'all', or None).

        Returns
        -------
        np.ndarray
            Confusion matrix.
        """
        y_true, y_pred, _, _ = self.get_processed_data(y_true, y_pred)
        matrix, _, _ = cu.calculate_confusion_matrix(y_true, y_pred, labels, normalize)
        return matrix

    def roc_auc_score(self, y_true=None, y_pred=None, average="macro", **kwargs):
        """
        Compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

        Parameters
        ----------
        y_true : array-like, optional
            Ground truth (correct) target values.
        y_pred : array-like, optional
            Estimated probabilities or decision function.
        average : str, optional
            Averaging method ('macro', 'weighted', or None).

        Returns
        -------
        float or dict
            ROC AUC score.
        """
        y_true, y_score, binary, _ = self.get_processed_data2(y_true, y_pred)
        # 1. Only 1 class in y_true
        if len(np.unique(y_true)) == 1:
            raise ValueError("Only one class present in y_true. ROC AUC score is not defined in that case.")

        trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        # 2. Binary cases
        if binary or len(np.unique(y_true)) == 2:
            if y_score.ndim == 2:
                if y_score.shape[1] == 2:
                    y_score = y_score[:, 1]  #  probability of class Positive
                elif y_score.shape[1] == 1:
                    y_score = y_score.ravel()
                else:
                    raise ValueError(f"Target is binary but y_score has {y_score.shape[1]} columns.")
            tpr, fpr, _ = cu.calculate_roc_curve(y_true, y_score)
            return float(trapz(tpr, fpr))

        # 3. Multiclass (One-vs-Rest)
        classes = np.unique(y_true).tolist()
        auc_list = [float(trapz(*cu.calculate_roc_curve(np.where(y_true == cls, 1, 0), y_score[:, i])[:2]))
                    for i, cls in enumerate(classes)]

        if average == "macro":
            return float(np.mean(auc_list))
        if average == "weighted":
            weights = cu.calculate_class_support(y_true)
            return float(np.dot(weights, auc_list) / np.sum(weights))
        return dict(zip(classes, auc_list))

    def gini_index(self, y_true=None, y_pred=None, **kwargs):
        """
        Compute the Gini index based on the ROC AUC score.

        Parameters
        ----------
        y_true : array-like, optional
            Ground truth (correct) target values.
        y_pred : array-like, optional
            Estimated probabilities or decision function.

        Returns
        -------
        float or dict
            Gini index.
        """
        auc_val = self.roc_auc_score(y_true, y_pred, **kwargs)
        return {k: 2 * v - 1.0 for k, v in auc_val.items()} if isinstance(auc_val, dict) else float(2 * auc_val - 1.0)

    def brier_score_loss(self, y_true=None, y_pred=None, **kwargs):
        """
        Parameters
        ----------
        y_true : array-like, optional
            Ground truth (correct) target values.
        y_pred : array-like, optional
            Predicted probabilities.

        Returns
        -------
        float
            Brier score loss.
        """
        y_true, y_pred, _, _ = self.get_processed_data2(y_true, y_pred)
        return float(np.mean(np.sum((np.eye(y_pred.shape[1] if y_pred.ndim > 1 else 2)[y_true.astype(int)] - y_pred) ** 2, axis=1)))

    def crossentropy_loss(self, y_true=None, y_pred=None, **kwargs):
        """
        Parameters
        ----------
        y_true : array-like, optional
            Ground truth (correct) target values.
        y_pred : array-like, optional
            Predicted probabilities.

        Returns
        -------
        float
            Cross-entropy loss.
        """
        y_true, y_pred, _, _ = self.get_processed_data2(y_true, y_pred)

        # 1. Transmit 1D hard labels [0, 2] or 2D soft labels [[0.9, 0.1]]
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            n_classes = y_pred.shape[1] if y_pred.ndim > 1 else 2
            y_t = np.eye(n_classes)[y_true.ravel().astype(int)]
        else:
            y_t = y_true.astype(float)
        # 2. ONLY the lower bound clip to avoid the log(0) trap, the upper bound 1.0 is absolutely safe.
        y_p = np.clip(y_pred, self.EPSILON, 1.0)
        return float(-np.mean(np.sum(y_t * np.log(y_p), axis=1)))

    def hinge_loss(self, y_true=None, y_pred=None, **kwargs):
        """
        Parameters
        ----------
        y_true : array-like, optional
            Ground truth (correct) target values.
        y_pred : array-like, optional
            Predicted scores.

        Returns
        -------
        float
            Hinge loss.
        """
        y_true, y_pred, _, _ = self.get_processed_data2(y_true, y_pred)
        y_true_oh = np.eye(y_pred.shape[1] if y_pred.ndim > 1 else 2)[y_true.astype(int)]
        return float(np.mean(np.maximum(0.0, np.max((1.0 - y_true_oh) * y_pred, axis=1) - np.sum(y_true_oh * y_pred, axis=1) + 1.0)))

    def kullback_leibler_divergence_loss(self, y_true=None, y_pred=None, **kwargs):
        """
        Parameters
        ----------
        y_true : array-like, optional
            Ground truth (correct) target values.
        y_pred : array-like, optional
            Predicted probabilities.

        Returns
        -------
        float
            Kullback-Leibler divergence loss.
        """
        y_true, y_pred, _, _ = self.get_processed_data2(y_true, y_pred)

        # 1. Pass Hard label [0, 2, 1] or pass Soft label [[0.8, 0.2]]
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            n_classes = y_pred.shape[1] if y_pred.ndim > 1 else 2
            y_t = np.eye(n_classes)[y_true.ravel().astype(int)]
        else:
            y_t = y_true.astype(float)
        # 2. Only clip y_pred to avoid log(0), preserve the purity of y_true.
        y_p = np.clip(y_pred, self.EPSILON, 1.0)
        # 3. Technique to eliminate the "0 * -inf = nan" trap:
        # Where y_t == 0, force the ratio y_t / y_p = 1.0 -> log(1.0) = 0 -> 0 * 0 = 0
        ratio = np.where(y_t > 0, y_t / y_p, 1.0)
        return float(np.mean(np.sum(y_t * np.log(ratio), axis=1)))


    CM = confusion_matrix
    PS = precision_score
    NPV = negative_predictive_value
    RS = recall_score
    AS = accuracy_score
    F1S = f1_score
    F2S = f2_score
    FBS = fbeta_score
    SS = specificity_score
    MCC = matthews_correlation_coefficient
    CKS = cohen_kappa_score
    ROC = AUC = RAS = roc_auc_score
    JSI = jaccard_similarity_coefficient = JSS = jaccard_similarity_score = JSC = jaccard_similarity_index
    GMS = g_mean_score
    GINI = gini_index
    LS = lift_score

    HML = hamming_loss
    HGL = hinge_loss
    KLDL = kullback_leibler_divergence_loss
    BSL = brier_score_loss
    CEL = crossentropy_loss
