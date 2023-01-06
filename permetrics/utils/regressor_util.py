#!/usr/bin/env python
# Created by "Thieu" at 12:23, 19/05/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


def calculate_nse(y_true, y_pred, one_dim):
    if one_dim:
        return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    else:
        return 1 - np.sum((y_true - y_pred) ** 2, axis=0) / np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)


def calculate_pcc(y_true, y_pred, one_dim):
    if one_dim:
        m1, m2 = np.mean(y_true), np.mean(y_pred)
        numerator = np.sum((y_true - m1) * (y_pred - m2))
        denominator = np.sqrt(np.sum((y_true - m1) ** 2)) * np.sqrt(np.sum((y_pred - m2) ** 2))
    else:
        m1, m2 = np.mean(y_true, axis=0), np.mean(y_pred, axis=0)
        numerator = np.sum((y_true - m1) * (y_pred - m2), axis=0)
        denominator = np.sqrt(np.sum((y_true - m1) ** 2, axis=0)) * np.sqrt(np.sum((y_pred - m2) ** 2, axis=0))
    return numerator / denominator


def calculate_absolute_pcc(y_true, y_pred, one_dim):
    if one_dim:
        m1, m2 = np.mean(y_true), np.mean(y_pred)
        numerator = np.sum(np.abs(y_true - m1) * np.abs(y_pred - m2))
        denominator = np.sqrt(np.sum((y_true - m1) ** 2)) * np.sqrt(np.sum((y_pred - m2) ** 2))
    else:
        m1, m2 = np.mean(y_true, axis=0), np.mean(y_pred, axis=0)
        numerator = np.sum(np.abs(y_true - m1) * np.abs(y_pred - m2), axis=0)
        denominator = np.sqrt(np.sum((y_true - m1) ** 2, axis=0)) * np.sqrt(np.sum((y_pred - m2) ** 2, axis=0))
    return numerator / denominator


def calculate_wi(y_true, y_pred, one_dim):
    if one_dim:
        m1 = np.mean(y_true)
        return 1 - np.sum((y_pred - y_true) ** 2) / np.sum((np.abs(y_pred - m1) + np.abs(y_true - m1)) ** 2)
    else:
        m1 = np.mean(y_true, axis=0)
        return 1 - np.sum((y_pred - y_true) ** 2, axis=0) / np.sum((np.abs(y_pred - m1) + np.abs(y_true - m1)) ** 2, axis=0)


def calculate_entropy(y_true, y_pred, one_dim):
    if one_dim:
        return -np.sum(y_true * np.log2(y_pred))
    else:
        return -np.sum(y_true * np.log2(y_pred), axis=0)


def calculate_ec(y_true, y_pred, one_dim):
    if one_dim:
        m1 = np.mean(y_true)
        numerator = np.sum((y_true - y_pred)**2)
        denominator = np.sum((y_true - m1)**2)
    else:
        m1 = np.mean(y_true, axis=0)
        numerator = np.sum((y_true - y_pred)**2, axis=0)
        denominator = np.sum((y_true - m1) ** 2, axis=0)
    return 1.0 - numerator / denominator


def calculate_mse(y_true, y_pred, one_dim):
    if one_dim:
        return np.mean((y_true - y_pred) ** 2)
    else:
        return np.mean((y_true - y_pred) ** 2, axis=0)
