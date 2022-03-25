#!/usr/bin/env python
# Created by "Thieu" at 10:48, 25/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from copy import deepcopy


class Evaluator:
    """
    This is base class for all performance metrics
    """

    EPSILON = 1e-10
    ACCEPTED_TYPE = (list, tuple, np.ndarray)

    def __init__(self, y_true=None, y_pred=None, decimal=5, **kwargs):
        """
        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            decimal (int): The number of fractional parts after the decimal point
        """
        if kwargs is None: kwargs = {}
        self.set_keyword_arguments(kwargs)
        self.y_true_original = y_true
        self.y_pred_original = y_pred
        self.y_true = deepcopy(y_true)
        self.y_pred = deepcopy(y_pred)
        self.decimal = decimal
        self.y_true_clean, self.y_pred_clean = None, None
        self.one_dim, self.already_clean = False, False

    def set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __clean_data(self, y_true=None, y_pred=None):
        """
        Get clean data and additional information for latter use

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values

        Returns:
            y_true: after remove all Nan and Inf values
            y_pred: after remove all Nan and Inf values
            y_true_clean: after remove all Nan, Inf and 0 values
            y_pred_clean: after remove all Nan, Inf and 0 values
            one_dim: is y_true has 1 dimensions or not
        """
        if isinstance(y_true, self.ACCEPTED_TYPE) and isinstance(y_pred, self.ACCEPTED_TYPE):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            ## Remove all dimensions of size 1
            y_true, y_pred = np.squeeze(y_true), np.squeeze(y_pred)
            # x = x[~np.isnan(x)] can't remove if array is dtype object, only work with dtype float
            y_true = y_true.astype('float64')
            y_pred = y_pred.astype('float64')

            if y_true.ndim == y_pred.ndim == 1:
                ## Remove all Nan in y_pred
                y_true = y_true[~np.isnan(y_pred)]
                y_pred = y_pred[~np.isnan(y_pred)]
                ## Remove all Inf in y_pred
                y_true = y_true[np.isfinite(y_pred)]
                y_pred = y_pred[np.isfinite(y_pred)]
                y_true_clean = y_true[y_pred != 0]
                y_pred_clean = y_pred[y_pred != 0]
                return y_true, y_pred, y_true_clean, y_pred_clean, True
            elif y_true.ndim == y_pred.ndim > 1:
                ## Remove all row with Nan in y_pred
                y_true = y_true[~np.isnan(y_pred).any(axis=1)]
                y_pred = y_pred[~np.isnan(y_pred).any(axis=1)]
                ## Remove all row with Inf in y_pred
                y_true = y_true[np.isfinite(y_pred).all(axis=1)]
                y_pred = y_pred[np.isfinite(y_pred).all(axis=1)]
                y_true_clean = y_true[~np.any(y_pred == 0, axis=1)]
                y_pred_clean = y_pred[~np.any(y_pred == 0, axis=1)]
                return y_true, y_pred, y_true_clean, y_pred_clean, False
            else:
                print("Permetrics Error! y_true and y_pred need to have same number of dimensions.")
                exit(0)
        else:
            print("Permetrics Error! y_true and y_pred need to be a list, tuple or np.array.")
            exit(0)

    def __positive_data(self, y_true=None, y_pred=None, one_dim=False, positive_only=False):
        """
        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            one_dim (bool): is y_true has 1 dimensions or not
            positive_only (bool): Calculate metric based on positive values only or not.

        Returns:
            y_true_used: y_true with all positive values in computation process.
            y_pred_used: y_pred with all positive values in computation process
        """
        if not positive_only:
            return y_true, y_pred
        else:
            if one_dim:
                y_true_positive = y_true[y_pred > 0]
                y_pred_positive = y_pred[y_pred > 0]
                return y_true_positive, y_pred_positive
            else:
                y_true_positive = y_true[np.all(y_pred > 0, axis=1)]
                y_pred_positive = y_pred[np.all(y_pred > 0, axis=1)]
                return y_true_positive, y_pred_positive

    def __get_used_data(self, clean, y_true, y_pred, y_true_clean, y_pred_clean, one_dim):
        if clean:
            return y_true_clean, y_pred_clean, one_dim
        else:
            return y_true, y_pred, one_dim

    def get_clean_data(self, y_true=None, y_pred=None, clean=False):
        """
        Get the cleaned data, the data pass to function will have higher priority than data pass to class object

        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred)

        Returns:
            y_true_used: y_true used in computation process.
            y_pred_used: y_pred used in computation process
            one_dim: is y_true has 1 dimensions or not
        """
        if y_true is not None and y_pred is not None:
            self.y_true_original, self.y_pred_original = deepcopy(y_true), deepcopy(y_pred)
            self.y_true, self.y_pred, self.y_true_clean, self.y_pred_clean, self.one_dim = self.__clean_data(y_true, y_pred)
            return self.__get_used_data(clean, self.y_true, self.y_pred, self.y_true_clean, self.y_pred_clean, self.one_dim)
        else:
            if self.y_true is not None and self.y_pred is not None:
                if self.already_clean:
                    return self.__get_used_data(clean, self.y_true, self.y_pred, self.y_true_clean, self.y_pred_clean, self.one_dim)
                else:
                    self.y_true, self.y_pred, self.y_true_clean, self.y_pred_clean, self.one_dim = self.__clean_data(self.y_true, self.y_pred)
                    self.already_clean = True
                    return self.__get_used_data(clean, self.y_true, self.y_pred, self.y_true_clean, self.y_pred_clean, self.one_dim)
            else:
                print("Permetrics Error! You need to pass y_true and y_pred to object creation or function called.")
                exit(0)

    def get_preprocessed_data(self, y_true=None, y_pred=None, clean=False, decimal=None, positive_only=False):
        """
        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            clean (bool): Remove all rows contain 0 value in y_pred (some methods have denominator is y_pred)
            decimal (int): The number of fractional parts after the decimal point
            positive_only (bool): Calculate metric based on positive values only or not.

        Returns:
            y_true_final: y_true used in evaluation process.
            y_pred_final: y_pred used in evaluation process
            one_dim: is y_true has 1 dimensions or not
            decimal: The number of fractional parts after the decimal point
        """
        y_true, y_pred, one_dim = self.get_clean_data(y_true, y_pred, clean)
        y_true, y_pred = self.__positive_data(y_true, y_pred, one_dim, positive_only)
        decimal = self.decimal if decimal is None else decimal
        return y_true, y_pred, one_dim, decimal

    def get_multi_output_result(self, result=None, multi_output=None, decimal=None):
        """
        Get multiple output results based on selected parameter

        Args:
            result: The raw result from metric
            multi_output: "raw_values" - return multi-output, [weights] - return single output based on weights, else - return mean result
            decimal (int): The number of fractional parts after the decimal point

        Returns:
            final_result: Multiple outputs results based on selected parameter
        """
        if isinstance(multi_output, (tuple, list, set, np.ndarray)):
            weights = np.array(multi_output)
            if self.y_true.shape[1] != len(weights):
                print("Permetrics Error! Multi-output weights has different length with y_true")
                exit(0)
            return np.round(np.dot(result, multi_output), decimal)
        elif multi_output == "raw_values":  # Default: raw_values
            return np.round(result, decimal)
        else:
            return np.round(np.mean(result), decimal)
