#!/usr/bin/env python
# Created by "Thieu" at 10:48, 25/03/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from permetrics.utils import constant as co


class Evaluator:
    """
    This is base class for all performance metrics
    """

    EPSILON = co.EPSILON
    SUPPORT = {}

    def __init__(self, y_true=None, y_pred=None, **kwargs):
        """
        Args:
            y_true (tuple, list, np.ndarray): The ground truth values
            y_pred (tuple, list, np.ndarray): The prediction values
        """
        if kwargs is None: kwargs = {}
        self.set_keyword_arguments(kwargs)
        self.y_true = y_true
        self.y_pred = y_pred

    def set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_processed_data(self, y_true=None, y_pred=None):
        pass

    def get_output_result(self, result=None, n_out=None, multi_output=None, force_finite=None, finite_value=None):
        """
        Get final output result based on selected parameter

        Args:
            result: The raw result from metric
            n_out: The number of column in y_true or y_pred
            multi_output: `raw_values` - return multi-output, `weights` - return single output based on weights, else - return mean result
            force_finite: Make result as finite number
            finite_value: The value that used to replace the infinite value or NaN value.

        Returns:
            final_result: Final output results based on selected parameter
        """
        if force_finite:
            result[np.isnan(result) | np.isinf(result)] = finite_value
        if n_out == 1:
            return result.flatten()[0]
        if isinstance(multi_output, (tuple, list, set, np.ndarray)):
            weights = np.asarray(multi_output, dtype=float)
            if n_out != len(weights):
                raise ValueError("Multi-output weights must have the same length with y_true or y_pred!")
            return np.dot(result, multi_output)
        elif multi_output == "raw_values":  # Default: raw_values
            return result
        else:
            return np.mean(result)

    def get_metric_by_name(self, metric_name=str, paras=None) -> dict:
        """
        Get single metric by name, specific parameter of metric by dictionary

        Args:
            metric_name (str): Select name of metric
            paras (dict): Dictionary of hyper-parameter for that metric

        Returns:
            result (dict): { metric_name: value }
        """
        result = {}
        obj = getattr(self, metric_name)
        result[metric_name] = obj() if paras is None else obj(**paras)
        return result

    def get_metrics_by_list_names(self, list_metric_names=list, list_paras=None) -> dict:
        """
        Get results of list metrics by its name and parameters

        Args:
            list_metric_names (list): e.g, ["RMSE", "MAE", "MAPE"]
            list_paras (list): e.g, [ {"decimal": 5, None}, {"decimal": 4, "multi_output": "raw_values"}, {"decimal":6, "multi_output": [2, 3]} ]

        Returns:
            results (dict): e.g, { "RMSE": 0.25, "MAE": [0.3, 0.6], "MAPE": 0.15 }
        """
        results = {}
        for idx, metric_name in enumerate(list_metric_names):
            obj = getattr(self, metric_name)
            if list_paras is None:
                results[metric_name] = obj()
            else:
                if len(list_metric_names) != len(list_paras):
                    raise ValueError("list_metric_names and list_paras must have the same length!")
                if list_paras[idx] is None:
                    results[metric_name] = obj()
                else:
                    results[metric_name] = obj(**list_paras[idx])
        return results

    def get_metrics_by_dict(self, metrics_dict: dict) -> dict:
        """
        Get results of list metrics by its name and parameters wrapped by dictionary

        For example:
            {"RMSE": { "multi_output": multi_output, "decimal": 4 }, "MAE": { "non_zero": True, "multi_output": multi_output, "decimal": 6}}

        Args:
            metrics_dict (dict): key is metric name and value is dict of parameters

        Returns:
            results (dict): e.g, { "RMSE": 0.3524, "MAE": 0.445263 }
        """
        results = {}
        for metric_name, paras_dict in metrics_dict.items():
            obj = getattr(self, metric_name)
            if paras_dict is None:
                results[metric_name] = obj()
            else:
                results[metric_name] = obj(**paras_dict)  # Unpacking a dictionary and passing it to function
        return results
