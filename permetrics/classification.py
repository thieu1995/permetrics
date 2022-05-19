# !/usr/bin/env python
# Created by "Thieu" at 09:29, 23/09/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from permetrics.evaluator import Evaluator
from permetrics.utils.data_util import *
from permetrics.utils.classifier_util import *
import numpy as np


class ClassificationMetric(Evaluator):
    """
    This is class contains all classification metrics (for both binary and multiple classification problem)

    Notes
    ~~~~~
    + Extension of: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
    """

    def __init__(self, y_true=None, y_pred=None, decimal=5, **kwargs):
        """
        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            decimal (int): The number of fractional parts after the decimal point
            **kwargs ():
        """
        super().__init__(y_true, y_pred, decimal, **kwargs)
        if kwargs is None: kwargs = {}
        self.set_keyword_arguments(kwargs)
        self.binary = True
        self.representor = "number"     # "number" or "string"

    def get_processed_data(self, y_true=None, y_pred=None, decimal=None):
        """
        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred)
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            y_true_final: y_true used in evaluation process.
            y_pred_final: y_pred used in evaluation process
            one_dim: is y_true has 1 dimensions or not
            decimal: The number of fractional parts after the decimal point
        """
        decimal = self.decimal if decimal is None else decimal
        if (y_true is not None) and (y_pred is not None):
            y_true, y_pred = format_classification_data_type(y_true, y_pred)
            y_true, y_pred, binary, representor = format_classification_data(y_true, y_pred)
        else:
            if (self.y_true is not None) and (self.y_pred is not None):
                y_true, y_pred = format_classification_data_type(self.y_true, self.y_pred)
                y_true, y_pred, binary, representor = format_classification_data(y_true, y_pred)
            else:
                print("Permetrics Error! You need to pass y_true and y_pred to object creation or function called.")
                exit(0)
        return y_true, y_pred, binary, representor, decimal

    def precision_score(self, y_true=None, y_pred=None, labels=None, average=None, decimal=None):
        """
        Generate precision score for multiple classification problem

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default=None, others=None
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
        matrix, imap, imap_count = confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = calculate_single_label_metric(matrix, imap, imap_count)

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

    def negative_predictive_value(self, y_true=None, y_pred=None, labels=None, average=None, decimal=None):
        """
        Generate negative predictive value for multiple classification problem

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default=None, others=None
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            npv (float, dict): the negative predictive value
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = calculate_single_label_metric(matrix, imap, imap_count)

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

    def specificity_score(self, y_true=None, y_pred=None, labels=None, average=None, decimal=None):
        """
        Generate specificity score for multiple classification problem

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default=None, others=None
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            ss (float, dict): the specificity score
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = calculate_single_label_metric(matrix, imap, imap_count)

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

    def recall_score(self, y_true=None, y_pred=None, labels=None, average=None, decimal=None):
        """
        Generate recall score for multiple classification problem

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default=None, others=None
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            recall (float, dict): the recall score
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = calculate_single_label_metric(matrix, imap, imap_count)

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

    def accuracy_score(self, y_true=None, y_pred=None, labels=None, average=None, decimal=None):
        """
        Generate accuracy score for multiple classification problem

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default=None, others=None
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            accuracy (float, dict): the accuracy score
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = calculate_single_label_metric(matrix, imap, imap_count)

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
            accuracy = dict([(label, np.round(item["precision"], decimal)) for label, item in metrics.items()])
        return accuracy if type(accuracy) == dict else np.round(accuracy, decimal)

    def f1_score(self, y_true=None, y_pred=None, labels=None, average=None, decimal=None):
        """
        Generate f1 score for multiple classification problem

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default=None, others=None
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            f1 (float, dict): the f1 score
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = calculate_single_label_metric(matrix, imap, imap_count)

        list_f1 = np.array([item["f1"] for item in metrics.values()])
        list_weights = np.array([item["n_true"] for item in metrics.values()])

        if average == "micro":
            tp_global = np.sum(np.diag(matrix))
            fp_global = fn_global = np.sum(matrix) - tp_global
            precision = np.round(tp_global / (tp_global + fp_global), decimal)
            recall = tp_global / (tp_global + fn_global)
            f1 = (2 * precision * recall) / (precision + recall)
        elif average == "macro":
            f1 = np.mean(list_f1)
        elif average == "weighted":
            f1 = np.dot(list_weights, list_f1) / np.sum(list_weights)
        else:
            f1 = dict([(label, item["f1"]) for label, item in metrics.items()])
        return f1 if type(f1) == dict else np.round(f1, decimal)

    def f2_score(self, y_true=None, y_pred=None, labels=None, average=None, decimal=None):
        """
        Generate f2 score for multiple classification problem

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default=None, others=None
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            f2 (float, dict): the f2 score
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = calculate_single_label_metric(matrix, imap, imap_count)

        list_f2 = np.array([item["f1"] for item in metrics.values()])
        list_weights = np.array([item["n_true"] for item in metrics.values()])

        if average == "micro":
            tp_global = np.sum(np.diag(matrix))
            fp_global = fn_global = np.sum(matrix) - tp_global
            precision = np.round(tp_global / (tp_global + fp_global), decimal)
            recall = tp_global / (tp_global + fn_global)
            f2 = (5 * precision * recall) / (4 * precision + recall)
        elif average == "macro":
            f2 = np.mean(list_f2)
        elif average == "weighted":
            f2 = np.dot(list_weights, list_f2) / np.sum(list_weights)
        else:
            f2 = dict([(label, item["f2"]) for label, item in metrics.items()])
        return f2 if type(f2) == dict else np.round(f2, decimal)

    def fbeta_score(self, y_true=None, y_pred=None, beta=1.0, labels=None, average=None, decimal=None):
        """
        The beta parameter determines the weight of recall in the combined score.
        beta < 1 lends more weight to precision, while beta > 1 favors recall
        (beta -> 0 considers only precision, beta -> +inf only recall).

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            beta (float): the weight of recall in the combined score, default = 1.0
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default=None, others=None
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            fbeta (float, dict): the fbeta score
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = calculate_single_label_metric(matrix, imap, imap_count, beta=beta)

        list_fbeta = np.array([item["f1"] for item in metrics.values()])
        list_weights = np.array([item["n_true"] for item in metrics.values()])

        if average == "micro":
            tp_global = np.sum(np.diag(matrix))
            fp_global = fn_global = np.sum(matrix) - tp_global
            precision = np.round(tp_global / (tp_global + fp_global), decimal)
            recall = tp_global / (tp_global + fn_global)
            fbeta = ((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall)
        elif average == "macro":
            fbeta = np.mean(list_fbeta)
        elif average == "weighted":
            fbeta = np.dot(list_weights, list_fbeta) / np.sum(list_weights)
        else:
            fbeta = dict([(label, item["fbeta"]) for label, item in metrics.items()])
        return fbeta if type(fbeta) == dict else np.round(fbeta, decimal)

    def matthews_correlation_coefficient(self, y_true=None, y_pred=None, labels=None, average=None, decimal=None):
        """
        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            beta (float): the weight of recall in the combined score, default = 1.0
            average (str, None): {'micro', 'macro', 'weighted'} or None, default=None, others=None
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            mcc (float, dict): the Matthews correlation coefficient
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = calculate_single_label_metric(matrix, imap, imap_count)

        list_mcc = np.array([item["mcc"] for item in metrics.values()])
        list_weights = np.array([item["n_true"] for item in metrics.values()])

        if average == "micro":
            tp = tn = np.sum(np.diag(matrix))
            fp = fn = np.sum(matrix) - tp
            mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        elif average == "macro":
            mcc = np.mean(list_mcc)
        elif average == "weighted":
            mcc = np.dot(list_weights, list_mcc) / np.sum(list_weights)
        else:
            mcc = dict([(label, item["mcc"]) for label, item in metrics.items()])
        return mcc if type(mcc) == dict else np.round(mcc, decimal)

    def hamming_loss(self, y_true=None, y_pred=None, labels=None, average=None, decimal=None):
        """
        Generate hamming loss for multiple classification problem

        Args:
            y_true (tuple, list, np.ndarray): a list of integers or strings for known classes
            y_pred (tuple, list, np.ndarray): a list of integers or strings for y_pred classes
            labels (tuple, list, np.ndarray): List of labels to index the matrix. This may be used to reorder or select a subset of labels.
            average (str, None): {'micro', 'macro', 'weighted'} or None, default=None, others=None
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            hl (float, dict): the hamming loss
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        matrix, imap, imap_count = confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = calculate_single_label_metric(matrix, imap, imap_count)

        list_accuracy = np.array([item["accuracy"] for item in metrics.values()])
        list_weights = np.array([item["n_true"] for item in metrics.values()])
        list_tp = np.array([item['tp'] for item in metrics.values()])

        if average == "micro":
            hl = 1.0 - np.sum(list_tp) / np.sum(list_weights)
        elif average == "macro":
            hl = np.mean(list_accuracy)
        elif average == "weighted":
            hl = np.dot(list_weights, list_accuracy) / np.sum(list_weights)
        else:
            hl = dict([(label, np.round(item["hamming_loss"], decimal)) for label, item in metrics.items()])
        return hl if type(hl) == dict else np.round(hl, decimal)


    def mean_log_likelihood(self, y_true=None, y_pred=None, multi_output="raw_values", decimal=None, non_zero=True, positive=True):
        """
        Mean Log Likelihood (MLL): Best possible score is ..., the higher value is better. Range = (-inf, +inf)

        Link: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/elementwise.py#L235

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = True)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = True)

        Returns:
            result (float, int, np.ndarray): MLL metric
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 1)
        else:
            y_pred[y_pred == 0] = self.EPSILON
        y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        if one_dim:
            return np.round(np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))), decimal)
        else:
            result = np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)), axis=0)
            return self.get_multi_output_result(result, multi_output, decimal)

    def single_log_likelihood(self, y_true=None, y_pred=None, decimal=None, non_zero=True, positive=True):
        """
        Log Likelihood (LL): Best possible score is ..., the higher value is better. Range = (-inf, +inf)

        Note: Computes the log likelihood between two numbers, or for element between a pair of list, tuple or numpy arrays.

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            multi_output: Can be "raw_values" or list weights of variables such as [0.5, 0.2, 0.3] for 3 columns, (Optional, default = "raw_values")
            decimal (int): The number of fractional parts after the decimal point (Optional, default = 5)
            non_zero (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred) (Optional, default = True)
            positive (bool): Calculate metric based on positive values only or not (Optional, default = True)

        Returns:
            result (float, int, np.ndarray): LL metric
        """
        y_true, y_pred, one_dim, decimal = self.get_processed_data(y_true, y_pred, decimal)
        if non_zero:
            y_true, y_pred = self.get_non_zero_data(y_true, y_pred, one_dim, 1)
        else:
            y_pred[y_pred == 0] = self.EPSILON
        y_true, y_pred = self.get_positive_data(y_true, y_pred, one_dim, 2)
        return np.round(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)), decimal)

    MLL = mll = mean_log_likelihood
    LL = ll = single_log_likelihood

    PS = ps = precision_score
    NPV = npv = negative_predictive_value
    RS = rs = recall_score
    AS = accuracy_score
    F1S = f1s = f1_score
    F2S = f2s = f2_score
    FBS = fbs = fbeta_score
    SS = ss = specificity_score
    MCC = mcc = matthews_correlation_coefficient
    HL = hl = hamming_loss


