#!/usr/bin/env python
# Created by "Thieu" at 12:12, 19/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import copy as cp


def format_regression_data_type(y_true, y_pred):
    if isinstance(y_true, (list, tuple, np.ndarray)) and isinstance(y_pred, (list, tuple, np.ndarray)):
        ## Remove all dimensions of size 1
        y_true, y_pred = np.squeeze(np.asarray(y_true, dtype='float64')), np.squeeze(np.asarray(y_pred, dtype='float64'))
        if y_true.ndim == y_pred.ndim:
            return y_true, y_pred
        else:
            raise ValueError("y_true and y_pred must have the same number of dimensions.")
    else:
        raise TypeError("y_true and y_pred must be lists, tuples or numpy arrays.")


def format_regression_data(y_true: np.ndarray, y_pred: np.ndarray):
    if y_true.ndim == y_pred.ndim == 1:
        ## Remove all Nan in y_pred
        y_true = y_true[~np.isnan(y_pred)]
        y_pred = y_pred[~np.isnan(y_pred)]
        ## Remove all Inf in y_pred
        y_true = y_true[np.isfinite(y_pred)]
        y_pred = y_pred[np.isfinite(y_pred)]
        return y_true, y_pred, True
    elif y_true.ndim == y_pred.ndim > 1:
        ## Remove all row with Nan in y_pred
        y_true = y_true[~np.isnan(y_pred).any(axis=1)]
        y_pred = y_pred[~np.isnan(y_pred).any(axis=1)]
        ## Remove all row with Inf in y_pred
        y_true = y_true[np.isfinite(y_pred).all(axis=1)]
        y_pred = y_pred[np.isfinite(y_pred).all(axis=1)]
        return y_true, y_pred, False


def get_regression_non_zero_data(y_true, y_pred, one_dim=True, rule_idx=0):
    """
    Get non-zero data based on rule

    Args:
        y_true (tuple, list, np.ndarray): The ground truth values
        y_pred (tuple, list, np.ndarray): The prediction values
        one_dim (bool): is y_true has 1 dimensions or not
        rule_idx (int): valid values [0, 1, 2] corresponding to [y_true, y_pred, both true and pred]

    Returns:
        y_true: y_true with positive values based on rule
        y_pred: y_pred with positive values based on rule

    """
    if rule_idx == 0:
        y_rule = cp.deepcopy(y_true)
    elif rule_idx == 1:
        y_rule = cp.deepcopy(y_pred)
    else:
        if one_dim:
            y_true_non, y_pred_non = y_true[y_true != 0], y_pred[y_true != 0]
            y_true, y_pred = y_true_non[y_pred_non != 0], y_pred_non[y_pred_non != 0]
        else:
            y_true_non, y_pred_non = y_true[~np.any(y_true == 0, axis=1)], y_pred[~np.any(y_true == 0, axis=1)]
            y_true, y_pred = y_true_non[~np.any(y_pred_non == 0, axis=1)], y_pred_non[~np.any(y_pred_non == 0, axis=1)]
        return y_true, y_pred
    if one_dim:
        y_true, y_pred = y_true[y_rule != 0], y_pred[y_rule != 0]
    else:
        y_true, y_pred = y_true[~np.any(y_rule == 0, axis=1)], y_pred[~np.any(y_rule == 0, axis=1)]
    return y_true, y_pred


def get_regression_positive_data(y_true, y_pred, one_dim=True, rule_idx=0):
    """
    Get positive data based on rule

    Args:
        y_true (tuple, list, np.ndarray): The ground truth values
        y_pred (tuple, list, np.ndarray): The prediction values
        one_dim (bool): is y_true has 1 dimensions or not
        rule_idx (int): valid values [0, 1, 2] corresponding to [y_true, y_pred, both true and pred]

    Returns:
        y_true: y_true with positive values based on rule
        y_pred: y_pred with positive values based on rule
    """
    if rule_idx == 0:
        y_rule = cp.deepcopy(y_true)
    elif rule_idx == 1:
        y_rule = cp.deepcopy(y_pred)
    else:
        if one_dim:
            y_true_non, y_pred_non = y_true[y_true > 0], y_pred[y_true > 0]
            y_true, y_pred = y_true_non[y_pred_non > 0], y_pred_non[y_pred_non > 0]
        else:
            y_true_non, y_pred_non = y_true[np.all(y_true > 0, axis=1)], y_pred[np.all(y_true > 0, axis=1)]
            y_true, y_pred = y_true_non[np.all(y_pred_non > 0, axis=1)], y_pred_non[np.all(y_pred_non > 0, axis=1)]
        return y_true, y_pred
    if one_dim:
        y_true, y_pred = y_true[y_rule > 0], y_pred[y_rule > 0]
    else:
        y_true, y_pred = y_true[np.all(y_rule > 0, axis=1)], y_pred[np.all(y_rule > 0, axis=1)]
    return y_true, y_pred


def format_classification_data(y_true: np.ndarray, y_pred: np.ndarray):
    if not (isinstance(y_true, (list, tuple, np.ndarray)) and isinstance(y_pred, (list, tuple, np.ndarray))):
        raise TypeError("y_true and y_pred must be lists, tuples or numpy arrays.")
    else:
        ## Remove all dimensions of size 1
        y_true, y_pred = np.squeeze(np.asarray(y_true)), np.squeeze(np.asarray(y_pred))
        if not (y_true.ndim == y_pred.ndim):
            raise TypeError("y_true and y_pred must have the same number of dimensions.")
        else:
            if np.issubdtype(y_true.dtype, np.number):
                var_type = "number"
                if y_true.ndim > 1:
                    y_true, y_pred = y_true.argmax(axis=1), y_pred.argmax(axis=1)
            else:
                var_type = "string"
                if y_true.ndim > 1:
                    raise ValueError("y_true and y_pred have n_dim > 1 need to be a number.")
            unique_true, unique_pred = sorted(np.unique(y_true)), sorted(np.unique(y_pred))
            if not (len(unique_pred) <= len(unique_true) and np.isin(unique_pred, unique_true).all()):
                raise ValueError("Existed at least one new label in y_pred.")
            else:
                binary = len(unique_true) == 2
            return y_true, y_pred, binary, var_type

