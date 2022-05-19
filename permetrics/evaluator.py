#!/usr/bin/env python
# Created by "Thieu" at 10:48, 25/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import copy as cp


class Evaluator:
    """
    This is base class for all performance metrics
    """

    EPSILON = 1e-10

    def __init__(self, y_true=None, y_pred=None, decimal=5, **kwargs):
        """
        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
            decimal (int): The number of fractional parts after the decimal point
        """
        if kwargs is None: kwargs = {}
        self.set_keyword_arguments(kwargs)
        self.y_true_original = cp.deepcopy(y_true)
        self.y_pred_original = cp.deepcopy(y_pred)
        self.y_true = cp.deepcopy(y_true)
        self.y_pred = cp.deepcopy(y_pred)
        self.decimal = decimal

    def set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_processed_data(self, y_true=None, y_pred=None, decimal=None):
        pass

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
