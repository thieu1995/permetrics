#!/usr/bin/env python
# Created by "Thieu" at 12:12, 19/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%
#
# import numpy as np
# import copy as cp
# from permetrics.utils.encoder import LabelEncoder
# import permetrics.utils.constant as co
#
#
# def format_regression_data_type(y_true: np.ndarray, y_pred: np.ndarray):
#     if isinstance(y_true, co.SUPPORTED_LIST) and isinstance(y_pred, co.SUPPORTED_LIST):
#         ## Remove all dimensions of size 1
#         y_true, y_pred = np.squeeze(np.asarray(y_true, dtype='float64')), np.squeeze(np.asarray(y_pred, dtype='float64'))
#         if y_true.ndim == y_pred.ndim:
#             if y_true.ndim == 1:
#                 return y_true.reshape(-1, 1), y_pred.reshape(-1, 1), 1      # n_outputs
#             if y_true.ndim > 2:
#                 raise ValueError("y_true and y_pred must be 1D or 2D arrays.")
#             return y_true, y_pred, y_true.shape[1]      # n_outputs
#         else:
#             raise ValueError("y_true and y_pred must have the same number of dimensions.")
#     else:
#         raise TypeError("y_true and y_pred must be lists, tuples or numpy arrays.")
#
#
# def get_regression_non_zero_data(y_true, y_pred, one_dim=True, rule_idx=0):
#     """
#     Get non-zero data based on rule
#
#     Args:
#         y_true (tuple, list, np.ndarray): The ground truth values
#         y_pred (tuple, list, np.ndarray): The prediction values
#         one_dim (bool): is y_true has 1 dimensions or not
#         rule_idx (int): valid values [0, 1, 2] corresponding to [y_true, y_pred, both true and pred]
#
#     Returns:
#         y_true: y_true with positive values based on rule
#         y_pred: y_pred with positive values based on rule
#
#     """
#     if rule_idx == 0:
#         y_rule = cp.deepcopy(y_true)
#     elif rule_idx == 1:
#         y_rule = cp.deepcopy(y_pred)
#     else:
#         if one_dim:
#             y_true_non, y_pred_non = y_true[y_true != 0], y_pred[y_true != 0]
#             y_true, y_pred = y_true_non[y_pred_non != 0], y_pred_non[y_pred_non != 0]
#         else:
#             y_true_non, y_pred_non = y_true[~np.any(y_true == 0, axis=1)], y_pred[~np.any(y_true == 0, axis=1)]
#             y_true, y_pred = y_true_non[~np.any(y_pred_non == 0, axis=1)], y_pred_non[~np.any(y_pred_non == 0, axis=1)]
#         return y_true, y_pred
#     if one_dim:
#         y_true, y_pred = y_true[y_rule != 0], y_pred[y_rule != 0]
#     else:
#         y_true, y_pred = y_true[~np.any(y_rule == 0, axis=1)], y_pred[~np.any(y_rule == 0, axis=1)]
#     return y_true, y_pred
#
#
# def get_regression_positive_data(y_true, y_pred, one_dim=True, rule_idx=0):
#     """
#     Get positive data based on rule
#
#     Args:
#         y_true (tuple, list, np.ndarray): The ground truth values
#         y_pred (tuple, list, np.ndarray): The prediction values
#         one_dim (bool): is y_true has 1 dimensions or not
#         rule_idx (int): valid values [0, 1, 2] corresponding to [y_true, y_pred, both true and pred]
#
#     Returns:
#         y_true: y_true with positive values based on rule
#         y_pred: y_pred with positive values based on rule
#     """
#     if rule_idx == 0:
#         y_rule = cp.deepcopy(y_true)
#     elif rule_idx == 1:
#         y_rule = cp.deepcopy(y_pred)
#     else:
#         if one_dim:
#             y_true_non, y_pred_non = y_true[y_true > 0], y_pred[y_true > 0]
#             y_true, y_pred = y_true_non[y_pred_non > 0], y_pred_non[y_pred_non > 0]
#         else:
#             y_true_non, y_pred_non = y_true[np.all(y_true > 0, axis=1)], y_pred[np.all(y_true > 0, axis=1)]
#             y_true, y_pred = y_true_non[np.all(y_pred_non > 0, axis=1)], y_pred_non[np.all(y_pred_non > 0, axis=1)]
#         return y_true, y_pred
#     if one_dim:
#         y_true, y_pred = y_true[y_rule > 0], y_pred[y_rule > 0]
#     else:
#         y_true, y_pred = y_true[np.all(y_rule > 0, axis=1)], y_pred[np.all(y_rule > 0, axis=1)]
#     return y_true, y_pred
#
#
# def format_classification_data(y_true: np.ndarray, y_pred: np.ndarray):
#     if not (isinstance(y_true, co.SUPPORTED_LIST) and isinstance(y_pred, co.SUPPORTED_LIST)):
#         raise TypeError("y_true and y_pred must be lists, tuples or numpy arrays.")
#     else:
#         ## Remove all dimensions of size 1
#         y_true, y_pred = np.squeeze(np.asarray(y_true)), np.squeeze(np.asarray(y_pred))
#         if np.issubdtype(y_true.dtype, np.number):
#             if np.isnan(y_true).any() or np.isinf(y_true).any():
#                 raise ValueError(f"Invalid y_true. It contains NaN or Inf value.")
#         if np.issubdtype(y_pred.dtype, np.number):
#             if np.isnan(y_pred).any() or np.isinf(y_pred).any():
#                 raise ValueError(f"Invalid y_pred. It contains NaN or Inf value.")
#
#         if y_true.ndim == y_pred.ndim:
#             if np.issubdtype(y_true.dtype, np.number) and np.issubdtype(y_pred.dtype, np.number):
#                 var_type = "number"
#                 if y_true.ndim > 1:
#                     y_true, y_pred = y_true.argmax(axis=1), y_pred.argmax(axis=1)
#                 else:
#                     y_true, y_pred = np.round(y_true).astype(int), np.round(y_pred).astype(int)
#             elif np.issubdtype(y_true.dtype, str) and np.issubdtype(y_pred.dtype, str):
#                 var_type = "string"
#                 if y_true.ndim > 1:
#                     raise ValueError("y_true and y_pred with ndim > 1 need to have data type as number.")
#             else:
#                 raise TypeError(f"y_true and y_pred need to have the same data type. {y_true.dtype} != {y_pred.dtype}")
#             unique_true, unique_pred = sorted(np.unique(y_true)), sorted(np.unique(y_pred))
#             if len(unique_pred) <= len(unique_true) and np.isin(unique_pred, unique_true).all():
#                 binary = len(unique_true) == 2
#             else:
#                 raise ValueError(f"Invalid y_pred, existed at least one new label in y_pred.")
#             return y_true, y_pred, binary, var_type
#         else:
#             if np.issubdtype(y_true.dtype, np.number):
#                 if y_true.ndim == 1:
#                     if np.issubdtype(y_pred.dtype, np.number):
#                         y_pred = y_pred.argmax(axis=1)
#                         var_type = "number"
#                         binary = len(np.unique(y_true)) == 2
#                         return y_true, y_pred, binary, var_type
#                     else:
#                         raise TypeError("Invalid y_pred, it should have data type as numeric.")
#                 else:
#                     y_true = y_true.argmax(axis=1)
#                     if np.issubdtype(y_pred.dtype, np.number):
#                         var_type = "number"
#                         binary = len(np.unique(y_true)) == 2
#                         return y_true, y_pred, binary, var_type
#                     else:
#                         raise TypeError("Invalid y_pred, it should have data type as numeric.")
#             else:
#                 raise ValueError("y_true has ndim > 1 and data type is string. You need to convert y_true to 1-D vector.")
#
#
# def format_y_score(y_true: np.ndarray, y_score: np.ndarray):
#     if not (isinstance(y_true, co.SUPPORTED_LIST) and isinstance(y_score, co.SUPPORTED_LIST)):
#         raise TypeError("y_true and y_score must be lists, tuples or numpy arrays.")
#     else:
#         y_true, y_score = np.squeeze(np.asarray(y_true)), np.squeeze(np.asarray(y_score))
#         if np.issubdtype(y_true.dtype, np.number):
#             if np.isnan(y_true).any() or np.isinf(y_true).any():
#                 raise ValueError(f"Invalid y_true. It contains NaN or Inf value.")
#         if np.issubdtype(y_score.dtype, np.number):
#             if np.isnan(y_score).any() or np.isinf(y_score).any():
#                 raise ValueError(f"Invalid y_score. It contains NaN or Inf value.")
#
#         if y_true.ndim > 1:
#             if np.issubdtype(y_true.dtype, np.number):
#                 y_true = y_true.argmax(axis=1)
#             else:
#                 raise TypeError(f"Invalid y_true. Its data type should be number and its shape is 1D vector")
#         var_type = "string" if np.issubdtype(y_true.dtype, str) else "number"
#         binary = len(np.unique(y_true)) == 2
#         le = LabelEncoder()
#         y_true = le.fit_transform(y_true).ravel()
#
#         if np.issubdtype(y_score.dtype, str) and y_score.ndim == 1:
#             y_score = le.transform(y_score).ravel()
#             y_score = np.eye(np.unique(y_true).size)[y_score]
#             return y_true, y_score, binary, var_type
#         elif np.issubdtype(y_score.dtype, np.number):
#             if y_score.ndim == 1:
#                 y_score = le.transform(y_score).ravel()
#                 y_score = np.eye(np.unique(y_true).size)[y_score]
#                 return y_true, y_score, binary, var_type
#             elif y_score.ndim == 2:
#                 if len(np.unique(y_true)) == y_score.shape[1]:
#                     return y_true, y_score, binary, var_type
#                 else:
#                     raise TypeError(f"Invalid y_score. It should has the number of columns = {len(np.unique(y_true))}")
#             else:
#                 raise TypeError(f"Invalid y_score. It should has shape of 1 or 2 dimensions")
#         else:
#             raise TypeError(f"Invalid y_true and y_score. Y_true data type should be number and y_score data type should be 1-hot matrix.")
#
#
# def is_unique_labels_consecutive_and_start_zero(vector):
#     labels = np.sort(np.unique(vector))
#     if 0 in labels:
#         if np.all(np.diff(labels) == 1):
#             return True
#     return False
#
#
# def format_external_clustering_data(y_true: np.ndarray, y_pred: np.ndarray):
#     """
#     Need both of y_true and y_pred to format
#     """
#     if not (isinstance(y_true, co.SUPPORTED_LIST) and isinstance(y_pred, co.SUPPORTED_LIST)):
#         raise TypeError("To calculate external clustering metrics, y_true and y_pred must be lists, tuples or numpy arrays.")
#     else:
#         ## Remove all dimensions of size 1
#         y_true, y_pred = np.squeeze(np.asarray(y_true)), np.squeeze(np.asarray(y_pred))
#         if not (y_true.ndim == y_pred.ndim):
#             raise TypeError("To calculate external clustering metrics, y_true and y_pred must have the same number of dimensions.")
#         else:
#             if y_true.ndim == 1:
#                 if np.issubdtype(y_true.dtype, np.number):
#                     if is_unique_labels_consecutive_and_start_zero(y_true):
#                         return y_true, y_pred, None
#                 le = LabelEncoder()
#                 y_true = le.fit_transform(y_true)
#                 y_pred = le.transform(y_pred)
#                 return y_true, y_pred, le
#             else:
#                 raise TypeError("To calculate clustering metrics, y_true and y_pred must be a 1-D vector.")
#
#
# def format_internal_clustering_data(y_pred: np.ndarray):
#     if not (isinstance(y_pred, co.SUPPORTED_LIST)):
#         raise TypeError("To calculate internal clustering metrics, y_pred must be lists, tuples or numpy arrays.")
#     else:
#         ## Remove all dimensions of size 1
#         y_pred = np.squeeze(np.asarray(y_pred))
#         if y_pred.ndim == 1:
#             if np.issubdtype(y_pred.dtype, np.number):
#                 y_pred = np.round(y_pred).astype(int)
#                 if is_unique_labels_consecutive_and_start_zero(y_pred):
#                     return y_pred, None
#             le = LabelEncoder()
#             labels = le.fit_transform(y_pred)
#             return labels, le
#         else:
#             raise TypeError("To calculate clustering metrics, labels must be a 1-D vector.")


import copy as cp
import numpy as np
import permetrics.utils.constant as co
from permetrics.utils.encoder import LabelEncoder


def _to_numpy(data, name, allow_nan_inf=False):
    try:
        arr = np.asarray(data)
    except (ValueError, TypeError):
        raise TypeError(f"{name} must be a list, tuple, or numpy array.")

    if np.issubdtype(arr.dtype, np.number) and not allow_nan_inf:
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError(f"{name} contains NaN or Inf.")
    return arr


def format_regression_data_type(y_true, y_pred):
    def _parse(data, name):
        arr = _to_numpy(data, name).astype(np.float64)
        if arr.ndim == 0:
            return arr.reshape(1)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr.ravel()
        if arr.ndim in (1, 2):
            return arr
        raise ValueError(f"{name} must be 1D or 2D. Got {arr.ndim}D.")

    yt = _parse(y_true, "y_true")
    yp = _parse(y_pred, "y_pred")

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}.")

    n_outputs = 1 if yt.ndim == 1 else yt.shape[1]
    return yt, yp, n_outputs


def _filter_regression_mask(y_true, y_pred, condition_func, rule_idx=0):
    if rule_idx == 0:
        mask = condition_func(y_true)
    elif rule_idx == 1:
        mask = condition_func(y_pred)
    elif rule_idx == 2:
        mask = condition_func(y_true) & condition_func(y_pred)
    else:
        raise ValueError("rule_idx must be 0 (y_true), 1 (y_pred), or 2 (both).")

    if y_true.ndim == 2:
        row_mask = np.all(mask, axis=1)
        return y_true[row_mask], y_pred[row_mask]

    return y_true[mask], y_pred[mask]


def get_regression_non_zero_data(y_true, y_pred, one_dim=None, rule_idx=0):
    return _filter_regression_mask(y_true, y_pred, lambda x: x != 0, rule_idx)


def get_regression_positive_data(y_true, y_pred, one_dim=None, rule_idx=0):
    return _filter_regression_mask(y_true, y_pred, lambda x: x > 0, rule_idx)


# ==========================================
# 2. CLASSIFICATION & PROBABILITY SCORES
# ==========================================


def _clean_class_array(arr, name):
    """Gỡ bỏ 'Pyramid of Doom', gom chung logic xử lý mảng argmax/one-hot"""
    arr = _to_numpy(arr, name)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim == 2:
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(f"2D matrix {name} must be numeric (one-hot/probs).")
        arr = arr.argmax(axis=1)
    elif arr.ndim > 2:
        raise ValueError(f"{name} must be 1D or 2D.")

    if np.issubdtype(arr.dtype, np.number):
        arr = np.round(arr).astype(np.int64)
        var_type = "number"
    elif np.issubdtype(arr.dtype, np.str_):
        var_type = "string"
    else:
        raise TypeError(f"Unsupported dtype {arr.dtype} for {name}.")

    return arr, var_type


def format_classification_data(y_true, y_pred):
    yt, type_t = _clean_class_array(y_true, "y_true")
    yp, type_p = _clean_class_array(y_pred, "y_pred")

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: {yt.shape} vs {yp.shape}.")
    if type_t != type_p:
        raise TypeError(f"Dtype mismatch: y_true ({type_t}) != y_pred ({type_p}).")

    u_true = np.unique(yt)
    if not np.isin(np.unique(yp), u_true).all():
        raise ValueError("y_pred contains unknown labels not present in y_true.")

    return yt, yp, len(u_true) == 2, type_t


def format_y_score(y_true, y_score):
    yt, type_t = _clean_class_array(y_true, "y_true")
    ys = _to_numpy(y_score, "y_score")

    le = LabelEncoder()
    yt_encoded = le.fit_transform(yt).ravel()
    n_classes = len(le.classes_)
    binary = n_classes == 2

    if ys.ndim == 1:
        # VÁ BUG NẶNG: Bung mảng xác suất Float 1D thành Matrix 2D cho Binary Class
        if np.issubdtype(ys.dtype, np.floating) and binary:
            ys_matrix = np.vstack([1.0 - ys, ys]).T
            return yt_encoded, ys_matrix, binary, type_t

        try:
            ys_transformed = le.transform(ys).ravel()
            ys_matrix = np.eye(n_classes)[ys_transformed]
        except Exception as e:
            raise ValueError(f"Failed to encode 1D y_score. Details: {e}")
        return yt_encoded, ys_matrix, binary, type_t

    elif ys.ndim == 2:
        if ys.shape[1] != n_classes:
            raise ValueError(
                f"y_score has {ys.shape[1]} cols, but y_true has {n_classes} classes."
            )
        return yt_encoded, ys, binary, type_t

    raise ValueError("y_score must be 1D or 2D.")


# ==========================================
# 3. CLUSTERING
# ==========================================


def is_unique_labels_consecutive_and_start_zero(vector):
    """Kiểm tra O(1) toán học thay vì ngồi đếm diff"""
    if not np.issubdtype(vector.dtype, np.integer):
        return False
    u = np.unique(vector)
    return len(u) > 0 and u[0] == 0 and u[-1] == len(u) - 1


def format_external_clustering_data(y_true, y_pred):
    yt = _to_numpy(y_true, "y_true").ravel()
    yp = _to_numpy(y_pred, "y_pred").ravel()

    if yt.shape != yp.shape:
        raise ValueError(f"Clustering shape mismatch: {yt.shape} vs {yp.shape}.")

    le_true = LabelEncoder()
    yt_clean = le_true.fit_transform(yt)

    # VÁ BUG GÃY KEY: Encode độc lập y_pred
    le_pred = LabelEncoder()
    yp_clean = le_pred.fit_transform(yp)

    return yt_clean, yp_clean, le_true


def format_internal_clustering_data(y_pred):
    yp = _to_numpy(y_pred, "y_pred").ravel()

    if is_unique_labels_consecutive_and_start_zero(yp):
        return yp.astype(np.int64), None

    le = LabelEncoder()
    labels = le.fit_transform(yp)
    return labels, le
