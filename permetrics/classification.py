# !/usr/bin/env python
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
    Defines a ClassificationMetric class that hold all classification metrics
    (for both binary and multiple classification problem)

    Parameters
    ----------
    y_true: tuple, list, np.ndarray, default = None
        The ground truth values.

    y_pred: tuple, list, np.ndarray, default = None
        The prediction values.

    decimal: int, default = 5
        The number of fractional parts after the decimal point

    labels: tuple, list, np.ndarray, default = None
        List of labels to index the matrix. This may be used to reorder or select a subset of labels.

    average: (str, None): {'micro', 'macro', 'weighted'} or None, default="macro"
        If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by considering each element of the label indicator matrix as a label.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label).
    """

    SUPPORT = {
        "PS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "NPV": {"type": "max", "range": "[0, 1]", "best": "1"},
        "RS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "AS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "F1S": {"type": "max", "range": "[0, 1]", "best": "1"},
        "F2S": {"type": "max", "range": "[0, 1]", "best": "1"},
        "FBS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "SS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "MCC": {"type": "max", "range": "[-1, +1]", "best": "1"},
        "HS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "CKS": {"type": "max", "range": "[-1, +1]", "best": "1"},
        "JSI": {"type": "max", "range": "[0, 1]", "best": "1"},
        "GMS": {"type": "max", "range": "[0, 1]", "best": "1"},
        "ROC-AUC": {"type": "max", "range": "[0, 1]", "best": "1"},
        "LS": {"type": "max", "range": "[0, +inf)", "best": "no best"},
        "GINI": {"type": "min", "range": "[0, 1]", "best": "0"},
        "CEL": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "HL": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "KLDL": {"type": "min", "range": "[0, +inf)", "best": "0"},
        "BSL": {"type": "min", "range": "[0, 1]", "best": "0"}
    }

    def __init__(self, y_true=None, y_pred=None, decimal=5, **kwargs):
        super().__init__(y_true, y_pred, decimal, **kwargs)
        if kwargs is None: kwargs = {}
        self.set_keyword_arguments(kwargs)
        self.binary = True
        self.representor = "number"     # "number" or "string"
        self.le = None  # LabelEncoder

    @staticmethod
    def get_support(name=None, verbose=True):
        if name == "all":
            if verbose:
                for key, value in ClassificationMetric.SUPPORT.items():
                    print(f"Metric {key} : {value}")
            return ClassificationMetric.SUPPORT
        if name not in list(ClassificationMetric.SUPPORT.keys()):
            raise ValueError(f"ClassificationMetric doesn't support metric named: {name}")
        else:
            if verbose:
                print(f"Metric {name}: {ClassificationMetric.SUPPORT[name]}")
            return ClassificationMetric.SUPPORT[name]

    def get_processed_data(self, y_true=None, y_pred=None, decimal=None):
        """
        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            decimal (int, None): The number of fractional parts after the decimal point

        Returns:
            y_true_final: y_true used in evaluation process.
            y_pred_final: y_pred used in evaluation process
            one_dim: is y_true has 1 dimensions or not
            decimal: The number of fractional parts after the decimal point
        """
        decimal = self.decimal if decimal is None else decimal
        if (y_true is not None) and (y_pred is not None):
            y_true, y_pred, binary, representor = du.format_classification_data(y_true, y_pred)
        else:
            if (self.y_true is not None) and (self.y_pred is not None):
                y_true, y_pred, binary, representor = du.format_classification_data(self.y_true, self.y_pred)
            else:
                raise ValueError("y_true or y_pred is None. You need to pass y_true and y_pred to object creation or function called.")
        return y_true, y_pred, binary, representor, decimal

    def get_processed_data2(self, y_true=None, y_pred=None, decimal=None):
        """
        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction scores
            decimal (int, None): The number of fractional parts after the decimal point

        Returns:
            y_true_final: y_true used in evaluation process.
            y_pred_final: y_pred used in evaluation process
            one_dim: is y_true has 1 dimensions or not
            decimal: The number of fractional parts after the decimal point
        """
        decimal = self.decimal if decimal is None else decimal
        if (y_true is not None) and (y_pred is not None):
            y_true, y_pred, binary, representor = du.format_y_score(y_true, y_pred)
        else:
            if (self.y_true is not None) and (self.y_pred is not None):
                y_true, y_pred, binary, representor = du.format_y_score(self.y_true, self.y_pred)
            else:
                raise ValueError("y_true or y_pred is None. You need to pass y_true and y_pred to object creation or function called.")
        return y_true, y_pred, binary, representor, decimal

    def confusion_matrix(self, y_true=None, y_pred=None, labels=None, normalize=None, **kwargs):
        """
        Generate confusion matrix and useful information

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
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal=None)
        matrix, imap, imap_count = cu.calculate_confusion_matrix(y_true, y_pred, labels, normalize)
        return matrix, imap, imap_count

    def precision_score(self, y_true=None, y_pred=None, labels=None, average="macro", decimal=None, **kwargs):
        """
        Generate precision score for multiple classification problem
        Higher is better (Best = 1), Range = [0, 1]

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default="macro"
                If None, the scores for each class are returned. Otherwise, this determines the type of averaging performed on the data:

                ``'micro'``:
                    Calculate metrics globally by considering each element of the label indicator matrix as a label.
                ``'macro'``:
                    Calculate metrics for each label, and find their unweighted mean.  This does not take label imbalance into account.
                ``'weighted'``:
                    Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label).

            decimal (int): The number of fractional parts after the decimal point

        Returns:
            precision (float, dict): the precision score
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = cu.calculate_confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = cu.calculate_single_label_metric(matrix, imap, imap_count)

        list_precision = np.array([item["precision"] for item in metrics.values()])
        list_weights = np.array([item["n_true"] for item in metrics.values()])

        if average == "micro":
            tp_global = np.sum(np.diag(matrix))
            fp_global = fn_global = np.sum(matrix) - tp_global
            precision = np.round(tp_global / (tp_global + fp_global), decimal)
        elif average == "macro":
            precision = np.mean(list_precision)
        elif average == "weighted":
            precision = np.dot(list_weights, list_precision) / np.sum(list_weights)
        else:
            precision = dict([(label, np.round(item["precision"], decimal)) for label, item in metrics.items()])
        return precision if type(precision) == dict else np.round(precision, decimal)

    def negative_predictive_value(self, y_true=None, y_pred=None, labels=None, average="macro", decimal=None, **kwargs):
        """
        Generate negative predictive value for multiple classification problem
        Higher is better (Best = 1), Range = [0, 1]

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default="macro"
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            npv (float, dict): the negative predictive value
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = cu.calculate_confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = cu.calculate_single_label_metric(matrix, imap, imap_count)

        list_npv = np.array([item["negative_predictive_value"] for item in metrics.values()])
        list_weights = np.array([item["n_true"] for item in metrics.values()])

        if average == "micro":
            tp_global = tn_global = np.sum(np.diag(matrix))
            fp_global = fn_global = np.sum(matrix) - tp_global
            npv = tn_global / (tn_global + fn_global)
        elif average == "macro":
            npv = np.mean(list_npv)
        elif average == "weighted":
            npv = np.dot(list_weights, list_npv) / np.sum(list_weights)
        else:
            npv = dict([(label, np.round(item["negative_predictive_value"], decimal)) for label, item in metrics.items()])
        return npv if type(npv) == dict else np.round(npv, decimal)

    def specificity_score(self, y_true=None, y_pred=None, labels=None, average="macro", decimal=None, **kwargs):
        """
        Generate specificity score for multiple classification problem
        Higher is better (Best = 1), Range = [0, 1]

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default="macro"
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            ss (float, dict): the specificity score
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = cu.calculate_confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = cu.calculate_single_label_metric(matrix, imap, imap_count)

        list_ss = np.array([item["specificity"] for item in metrics.values()])
        list_weights = np.array([item["n_true"] for item in metrics.values()])

        if average == "micro":
            tp_global = tn_global = np.sum(np.diag(matrix))
            fp_global = fn_global = np.sum(matrix) - tp_global
            ss = tn_global / (tn_global + fp_global)
        elif average == "macro":
            ss = np.mean(list_ss)
        elif average == "weighted":
            ss = np.dot(list_weights, list_ss) / np.sum(list_weights)
        else:
            ss = dict([(label, np.round(item["specificity"], decimal)) for label, item in metrics.items()])
        return ss if type(ss) == dict else np.round(ss, decimal)

    def recall_score(self, y_true=None, y_pred=None, labels=None, average="macro", decimal=None, **kwargs):
        """
        Generate recall score for multiple classification problem
        Higher is better (Best = 1), Range = [0, 1]

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default="macro"
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            recall (float, dict): the recall score
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = cu.calculate_confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = cu.calculate_single_label_metric(matrix, imap, imap_count)

        list_recall = np.array([item["recall"] for item in metrics.values()])
        list_weights = np.array([item["n_true"] for item in metrics.values()])

        if average == "micro":
            tp_global = np.sum(np.diag(matrix))
            fp_global = fn_global = np.sum(matrix) - tp_global
            recall = tp_global / (tp_global + fn_global)
        elif average == "macro":
            recall = np.mean(list_recall)
        elif average == "weighted":
            recall = np.dot(list_weights, list_recall) / np.sum(list_weights)
        else:
            recall = dict([(label, np.round(item["recall"], decimal)) for label, item in metrics.items()])
        return recall if type(recall) == dict else np.round(recall, decimal)

    def accuracy_score(self, y_true=None, y_pred=None, labels=None, average="macro", decimal=None, **kwargs):
        """
        Generate accuracy score for multiple classification problem
        Higher is better (Best = 1), Range = [0, 1]

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default="macro"
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            accuracy (float, dict): the accuracy score
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = cu.calculate_confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = cu.calculate_single_label_metric(matrix, imap, imap_count)

        list_accuracy = np.array([item["accuracy"] for item in metrics.values()])
        list_weights = np.array([item["n_true"] for item in metrics.values()])
        list_tp = np.array([item['tp'] for item in metrics.values()])

        if average == "micro":
            accuracy = np.sum(list_tp) / np.sum(list_weights)
        elif average == "macro":
            accuracy = np.mean(list_accuracy)
        elif average == "weighted":
            accuracy = np.dot(list_weights, list_accuracy) / np.sum(list_weights)
        else:
            accuracy = dict([(label, np.round(item["accuracy"], decimal)) for label, item in metrics.items()])
        return accuracy if type(accuracy) == dict else np.round(accuracy, decimal)

    def f1_score(self, y_true=None, y_pred=None, labels=None, average="macro", decimal=None, **kwargs):
        """
        Generate f1 score for multiple classification problem
        Higher is better (Best = 1), Range = [0, 1]

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default="macro"
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            f1 (float, dict): the f1 score
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = cu.calculate_confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = cu.calculate_single_label_metric(matrix, imap, imap_count)

        list_f1 = np.array([item["f1"] for item in metrics.values()])
        list_weights = np.array([item["n_true"] for item in metrics.values()])

        if average == "micro":
            tp_global = np.sum(np.diag(matrix))
            fp_global = fn_global = np.sum(matrix) - tp_global
            precision = tp_global / (tp_global + fp_global)
            recall = tp_global / (tp_global + fn_global)
            f1 = (2 * precision * recall) / (precision + recall)
        elif average == "macro":
            f1 = np.mean(list_f1)
        elif average == "weighted":
            f1 = np.dot(list_weights, list_f1) / np.sum(list_weights)
        else:
            f1 = dict([(label, np.round(item["f1"], decimal)) for label, item in metrics.items()])
        return f1 if type(f1) == dict else np.round(f1, decimal)

    def f2_score(self, y_true=None, y_pred=None, labels=None, average="macro", decimal=None, **kwargs):
        """
        Generate f2 score for multiple classification problem
        Higher is better (Best = 1), Range = [0, 1]

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default="macro"
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            f2 (float, dict): the f2 score
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = cu.calculate_confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = cu.calculate_single_label_metric(matrix, imap, imap_count)

        list_f2 = np.array([item["f2"] for item in metrics.values()])
        list_weights = np.array([item["n_true"] for item in metrics.values()])

        if average == "micro":
            tp_global = np.sum(np.diag(matrix))
            fp_global = fn_global = np.sum(matrix) - tp_global
            precision = tp_global / (tp_global + fp_global)
            recall = tp_global / (tp_global + fn_global)
            f2 = (5 * precision * recall) / (4 * precision + recall)
        elif average == "macro":
            f2 = np.mean(list_f2)
        elif average == "weighted":
            f2 = np.dot(list_weights, list_f2) / np.sum(list_weights)
        else:
            f2 = dict([(label, np.round(item["f2"], decimal)) for label, item in metrics.items()])
        return f2 if type(f2) == dict else np.round(f2, decimal)

    def fbeta_score(self, y_true=None, y_pred=None, beta=1.0, labels=None, average="macro", decimal=None, **kwargs):
        """
        The beta parameter determines the weight of recall in the combined score.
        beta < 1 lends more weight to precision, while beta > 1 favors recall
        (beta -> 0 considers only precision, beta -> +inf only recall).
        Higher is better (Best = 1), Range = [0, 1]

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            beta (float): the weight of recall in the combined score, default = 1.0
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default="macro"
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            fbeta (float, dict): the fbeta score
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = cu.calculate_confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = cu.calculate_single_label_metric(matrix, imap, imap_count, beta=beta)

        list_fbeta = np.array([item["fbeta"] for item in metrics.values()])
        list_weights = np.array([item["n_true"] for item in metrics.values()])

        if average == "micro":
            tp_global = np.sum(np.diag(matrix))
            fp_global = fn_global = np.sum(matrix) - tp_global
            precision = tp_global / (tp_global + fp_global)
            recall = tp_global / (tp_global + fn_global)
            fbeta = ((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall)
        elif average == "macro":
            fbeta = np.mean(list_fbeta)
        elif average == "weighted":
            fbeta = np.dot(list_weights, list_fbeta) / np.sum(list_weights)
        else:
            fbeta = dict([(label, np.round(item["fbeta"], decimal)) for label, item in metrics.items()])
        return fbeta if type(fbeta) == dict else np.round(fbeta, decimal)

    def matthews_correlation_coefficient(self, y_true=None, y_pred=None, labels=None, average="macro", decimal=None, **kwargs):
        """
        Generate Matthews Correlation Coefficient
        Higher is better (Best = 1), Range = [-1, +1]

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default="macro"
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            mcc (float, dict): the Matthews correlation coefficient
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = cu.calculate_confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = cu.calculate_single_label_metric(matrix, imap, imap_count)

        list_mcc = np.array([item["mcc"] for item in metrics.values()])
        list_weights = np.array([item["n_true"] for item in metrics.values()])

        if average == "micro":
            tp = tn = np.sum(np.diag(matrix))
            fp = fn = np.sum(matrix) - tp
            mcc = (tp * tn - fp * fn) / np.sqrt(((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        elif average == "macro":
            mcc = np.mean(list_mcc)
        elif average == "weighted":
            mcc = np.dot(list_weights, list_mcc) / np.sum(list_weights)
        else:
            mcc = dict([(label, np.round(item["mcc"], decimal)) for label, item in metrics.items()])
        return mcc if type(mcc) == dict else np.round(mcc, decimal)

    def hamming_score(self, y_true=None, y_pred=None, labels=None, average="macro", decimal=None, **kwargs):
        """
        Generate hamming score for multiple classification problem
        Higher is better (Best = 1), Range = [0, 1]

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default="macro"
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            hl (float, dict): the hamming score
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = cu.calculate_confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = cu.calculate_single_label_metric(matrix, imap, imap_count)

        list_hs = np.array([item["hamming_score"] for item in metrics.values()])
        list_weights = np.array([item["n_true"] for item in metrics.values()])
        list_tp = np.array([item['tp'] for item in metrics.values()])

        if average == "micro":
            hl = 1.0 - np.sum(list_tp) / np.sum(list_weights)
        elif average == "macro":
            hl = np.mean(list_hs)
        elif average == "weighted":
            hl = np.dot(list_weights, list_hs) / np.sum(list_weights)
        else:
            hl = dict([(label, np.round(item["hamming_score"], decimal)) for label, item in metrics.items()])
        return hl if type(hl) == dict else np.round(hl, decimal)

    def lift_score(self, y_true=None, y_pred=None, labels=None, average="macro", decimal=None, **kwargs):
        """
        Generate lift score for multiple classification problem
        Higher is better (Best = +1), Range = [0, +1]

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default="macro"
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            ls (float, dict): the lift score
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = cu.calculate_confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = cu.calculate_single_label_metric(matrix, imap, imap_count)

        list_ls = np.array([item["lift_score"] for item in metrics.values()])
        list_weights = np.array([item["n_true"] for item in metrics.values()])

        if average == "micro":
            tp = tn = np.sum(np.diag(matrix))
            fp = fn = np.sum(matrix) - tp
            ls = (tp/(tp + fp)) / ((tp + fn) / (tp + tn + fp + fn))
        elif average == "macro":
            ls = np.mean(list_ls)
        elif average == "weighted":
            ls = np.dot(list_weights, list_ls) / np.sum(list_weights)
        else:
            ls = dict([(label, np.round(item["lift_score"], decimal)) for label, item in metrics.items()])
        return ls if type(ls) == dict else np.round(ls, decimal)

    def cohen_kappa_score(self, y_true=None, y_pred=None, labels=None, average="macro", decimal=None, **kwargs):
        """
        Generate Cohen Kappa score for multiple classification problem
        Higher is better (Best = +1), Range = [-1, +1]

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default="macro"
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            cks (float, dict): the Cohen Kappa score
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = cu.calculate_confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = cu.calculate_single_label_metric(matrix, imap, imap_count)

        list_kp = np.array([item["kappa_score"] for item in metrics.values()])
        list_weights = np.array([item["n_true"] for item in metrics.values()])

        if average == 'weighted':
            kappa = np.dot(list_weights, list_kp) / np.sum(list_weights)
        elif average == 'macro':
            kappa = np.mean(list_kp)
        elif average == 'micro':
            kappa = np.average(list_kp)
        else:
            kappa = dict([(label, np.round(item["kappa_score"], decimal)) for label, item in metrics.items()])
        return kappa if type(kappa) == dict else np.round(kappa, decimal)

    def jaccard_similarity_index(self, y_true=None, y_pred=None, labels=None, average="macro", decimal=None, **kwargs):
        """
        Generate Jaccard similarity index for multiple classification problem
        Higher is better (Best = +1), Range = [0, +1]

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default="macro"
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            jsi (float, dict): the Jaccard similarity index
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = cu.calculate_confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = cu.calculate_single_label_metric(matrix, imap, imap_count)

        list_js = np.array([item["jaccard_similarities"] for item in metrics.values()])
        list_weights = np.array([item["n_true"] for item in metrics.values()])

        if average == "micro":
            tp = tn = np.sum(np.diag(matrix))
            fp = fn = np.sum(matrix) - tp
            js = tp / (tp + fp + fn)
        elif average == "macro":
            js = np.mean(list_js)
        elif average == "weighted":
            js = np.dot(list_weights, list_js) / np.sum(list_weights)
        else:
            js = dict([(label, np.round(item["jaccard_similarities"], decimal)) for label, item in metrics.items()])
        return js if type(js) == dict else np.round(js, decimal)

    def g_mean_score(self, y_true=None, y_pred=None, labels=None, average="macro", decimal=None, **kwargs):
        """
        Calculates the G-mean (Geometric mean) score between y_true and y_pred.
        Higher is better (Best = +1), Range = [0, +1]

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default="macro"
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            float, dict: The G-mean score.
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = cu.calculate_confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = cu.calculate_single_label_metric(matrix, imap, imap_count)

        list_gm = np.array([item["g_mean"] for item in metrics.values()])
        list_weights = np.array([item["n_true"] for item in metrics.values()])

        if average == "micro":
            tp = tn = np.sum(np.diag(matrix))
            fp = fn = np.sum(matrix) - tp
            gm = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))
        elif average == "macro":
            gm = np.mean(list_gm)
        elif average == "weighted":
            gm = np.dot(list_weights, list_gm) / np.sum(list_weights)
        else:
            gm = dict([(label, np.round(item["g_mean"], decimal)) for label, item in metrics.items()])
        return gm if type(gm) == dict else np.round(gm, decimal)

    def gini_index(self, y_true=None, y_pred=None, decimal=None, **kwargs):
        """
        Calculates the Gini index between y_true and y_pred.
        Smaller is better (Best = 0), Range = [0, +1]

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            average (str, None): {'macro', 'weighted'} or None, default="macro"
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            float, dict: The Gini index
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        # Calculate class probabilities
        total_samples = len(y_true)
        y_prob = np.zeros(total_samples)
        for idx in range(0, total_samples):
            if y_true[idx] == y_pred[idx]:
                y_prob[idx] = 1
            else:
                y_prob[idx] = 0
        positive_samples = np.sum(y_prob)
        negative_samples = total_samples - positive_samples
        p_positive = positive_samples / total_samples
        p_negative = negative_samples / total_samples
        # Calculate Gini index
        result = 1 - (p_positive ** 2 + p_negative ** 2)
        return np.round(result, decimal)

    def crossentropy_loss(self, y_true=None, y_pred=None, decimal=5, **kwargs):
        """
        Calculates the Cross-Entropy loss between y_true and y_pred.
        Smaller is better (Best = 0), Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): A LIST OF PREDICTED SCORES (NOT LABELS)
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            float: The Cross-Entropy loss
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data2(y_true, y_pred, decimal)
        if binary:
            y_pred = np.clip(y_pred, self.EPSILON, 1 - self.EPSILON)
            term_0 = (1 - y_true) * np.log(1 - y_pred)
            term_1 = y_true * np.log(y_pred)
            res = -np.mean(term_0 + term_1, axis=0)
            return np.round(res, decimal)
        else:
            # Convert y_true to one-hot encoded array
            num_classes = len(np.unique(y_true))
            y_true = np.eye(num_classes)[y_true]
            y_pred = np.clip(y_pred, self.EPSILON, 1 - self.EPSILON)
            res = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
            return np.round(res, decimal)

    def hinge_loss(self, y_true=None, y_pred=None, decimal=5, **kwargs):
        """
        Calculates the Hinge loss between y_true and y_pred.
        Smaller is better (Best = 0), Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of labels (or predicted scores in case of multi-class)
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            float: The Hinge loss
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data2(y_true, y_pred, decimal)
        if binary:
            # replacing 0 = -1
            y_true[y_true == 0] = -1
            y_pred[y_pred == 0] = -1
            res = np.mean([max(0, 1 - x * y) ** 2 for x, y in zip(y_true, y_pred)])
            return np.round(res, decimal)
        else:
            # Convert y_true to one-hot encoded array
            num_classes = len(np.unique(y_true))
            y_true = np.eye(num_classes)[y_true]
            neg = np.max((1 - y_true) * y_pred, axis=1)
            pos = np.sum(y_true * y_pred, axis=1)
            temp = neg - pos + 1
            temp[temp < 0] = 0
            return np.round(np.mean(temp), decimal)

    def kullback_leibler_divergence_loss(self, y_true=None, y_pred=None, decimal=5, **kwargs):
        """
        Calculates the Kullback-Leibler divergence loss between y_true and y_pred.
        Smaller is better (Best = 0), Range = [0, +inf)

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of labels (or predicted scores in case of multi-class)
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            float: The Kullback-Leibler divergence loss
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data2(y_true, y_pred, decimal)
        y_pred = np.clip(y_pred, self.EPSILON, 1 - self.EPSILON)  # Clip predicted probabilities
        if binary:
            y_true = np.clip(y_true, self.EPSILON, 1 - self.EPSILON)  # Clip true labels
            res = y_true * np.log(y_true / y_pred) + (1 - y_true) * np.log((1 - y_true) / (1 - y_pred))
            res = np.mean(res)
        else:
            # Convert y_true to one-hot encoded array
            num_classes = len(np.unique(y_true))
            y_true = np.eye(num_classes)[y_true]
            y_true = np.clip(y_true, self.EPSILON, 1 - self.EPSILON)  # Clip true labels
            res = np.sum(y_true * np.log(y_true / y_pred), axis=1)
            res = np.mean(res)
        return np.round(res, decimal)

    def brier_score_loss(self, y_true=None, y_pred=None, decimal=5, **kwargs):
        """
        Calculates the Brier Score Loss between y_true and y_pred.
        Smaller is better (Best = 0), Range = [0, 1]

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of labels (or predicted scores in case of multi-class)
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            float, dict: The Brier Score Loss
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data2(y_true, y_pred, decimal)
        if binary:
            res = np.mean((y_true - y_pred) ** 2)
        else:  # Multi-class classification
            # Convert y_true to one-hot encoded array
            num_classes = len(np.unique(y_true))
            y_true = np.eye(num_classes)[y_true]
            res = np.mean(np.sum((y_true - y_pred) ** 2, axis=1))
        return np.round(res, decimal)

    def roc_auc_score(self, y_true=None, y_pred=None, average="macro", decimal=5, **kwargs):
        """
        Calculates the ROC-AUC score between y_true and y_score.
        Higher is better (Best = +1), Range = [0, +1]

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): A LIST OF PREDICTED SCORES (NOT LABELS)
            average (str, None): {'macro', 'weighted'} or None, default="macro"
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            float, dict: The AUC score.
        """
        y_true, y_score, binary, representor, decimal = self.get_processed_data2(y_true, y_pred, decimal)
        if binary:
            tpr, fpr, thresholds = cu.calculate_roc_curve(y_true, y_score)
            # Calculate the area under the curve (AUC) using the trapezoidal rule
            return np.trapz(tpr, fpr)
        else:
            list_weights = cu.calculate_class_weights(y_true, y_pred=None, y_score=y_score)
            # one-vs-all (rest) approach
            tpr = dict()
            fpr = dict()
            thresholds = dict()
            auc = []
            n_classes = len(np.unique(y_true))
            for i in range(n_classes):
                y_true_i = np.array([1 if y == i else 0 for y in y_true])
                y_score_i = y_score[:, i]
                tpr[i], fpr[i], thresholds[i] = cu.calculate_roc_curve(y_true_i, y_score_i)
                # Calculate the area under the curve (AUC) using the trapezoidal rule
                auc.append(np.trapz(tpr[i], fpr[i]))
            if average == "macro":
                result = np.mean(auc)
            elif average == "weighted":
                result = np.dot(list_weights, auc) / np.sum(list_weights)
            else:
                result = dict([(idx, np.round(auc[idx], decimal)) for idx in range(n_classes)])
            return result if type(result) == dict else np.round(result, decimal)

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
    HS = hamming_score
    LS = lift_score
    CKS = cohen_kappa_score
    JSI = JSC = jaccard_similarity_coefficient = jaccard_similarity_index
    GMS = g_mean_score
    GINI = gini_index
    CEL = crossentropy_loss
    HL = hinge_loss
    KLDL = kullback_leibler_divergence_loss
    BSL = brier_score_loss
    ROC = AUC = RAS = roc_auc_score
