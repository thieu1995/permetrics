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
            precision(float, dict): the precision score
        """
        y_true, y_pred, binary, representor, decimal = self.get_processed_data(y_true, y_pred, decimal)

        matrix, imap, imap_count = confusion_matrix(y_true, y_pred, labels, normalize=None)
        metrics = calculate_single_label_metric(matrix, imap, imap_count)

        list_precision = np.array([item["precision"] for item in metrics.values()])
        list_recall = np.array([item["recall"] for item in metrics.values()])
        list_weights = np.array([item["n_true"] for item in metrics.values()])

        if average == "micro":
            tp_global = np.sum(np.diag(matrix))
            fp_global = fn_global = np.sum(matrix) - tp_global
            precision = np.round(tp_global / (tp_global + fp_global), decimal)
            recall = tp_global / (tp_global + fn_global)
        elif average == "macro":
            precision = np.mean(list_precision)
            recall = np.mean(list_recall)
        elif average == "weighted":
            precision = np.dot(list_weights, list_precision) / np.sum(list_weights)
            recall = np.dot(list_weights, list_recall) / np.sum(list_weights)
        else:
            precision, recall = {}, {}
            for label, item in metrics.items():
                precision[label] = item["precision"]
                recall[label] = item["recall"]
        return precision, recall


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
