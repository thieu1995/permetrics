#!/usr/bin/env python
# Created by "Thieu" at 09:58, 27/07/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import pytest

from permetrics import RegressionMetric


STANDARD_METRICS = [
    "EVS", "ME", "MBE", "MAE", "MSE", "RMSE", "MSLE", "MedAE", "MRE", "MPE", "MAPE",
    "SMAPE", "SMAPE_NP", "SMAPE_S", "SMAPE_S_P", "MAAPE", "MASE", "NSE", "NNSE", "WI",
    "R", "AR", "RSQ", "CI", "COD", "ACOD", "DRV", "KGE", "PCD", "CE", "KLD", "JSD",
    "VAF", "RAE", "RRSE", "A10", "A20", "A30", "NRMSE", "RSE", "COV", "COR",
    "EC", "OI", "CRM", "NGINI", "RGINI"
]

SAMPLE_WISE_METRICS = [ "RE", "AE", "SE", "SLE" ]


# Structure: (y_true, y_pred, ndim, weights_multi_output)
EDGE_CASES = [
    # 1. Standard case
    (np.array([3, -0.5, 2, 7, 5]), np.array([2.5, 0.0, 2, 8, 5]), 1, None),

    # 2. The input is a regular List instead of a NumPy Array (Check for cast type compatibility).
    ([3, -0.5, 2, 7, 5], (2.5, 0.0, 2, 8, 5), 1, None),

    # 3. Scalar values
    (5, 4, 1, None),

    # 4. Perfect Match
    (np.array([1.5, 2.0, 3.5]), np.array([1.5, 2.0, 3.5]), 1, None),

    # 5. Contains the value 0 (Very important for testing Division by Zero in metrics like MAPE)
    ([0.0, 2.0, 0.0, 4.0], np.array([0.1, 2.0, -0.1, 4.0]), 1, None),

    # 6. 2D Numpy Array (Multi-output) - 2 columns
    (
        np.array([[3, 1], [-0.5, 2], [2, 3], [7, 4], [5, 5], [6, 6]]),
        np.array([[2.5, 0.5], [0.0, 1.5], [2, 2], [8, 3], [5, 4], [3, 5]]),
        2,
        (0.3, 0.7)  # The weights must match the number of columns (in this case, 2).
    ),

    # 7. 2D Numpy Array - 3 columns (Further testing the flexibility of multi_output)
    (
        np.array([[1, 2, 3], [4, 5, 6]]),
        np.array([[1.1, 1.9, 3.2], [3.8, 5.1, 5.9]]),
        2,
        (0.2, 0.5, 0.3)  # 3 columns -> 3 weights
    ),

    # 8. Contains 0, negative number, and super small number
    ([10.0, -10.0, 1e-9], [0.0, 10.0, 0.1], 1, None),
]


@pytest.fixture(params=EDGE_CASES, scope="module")
def rm_data(request):
    """
    This fixture will automatically run through ALL the edge cases defined above.
    """
    y_true, y_pred, dims, weights = request.param
    rm = RegressionMetric(y_true=y_true, y_pred=y_pred)
    return rm, dims, weights


@pytest.mark.parametrize("metric_name", STANDARD_METRICS)
def test_regression_metrics(rm_data, metric_name):
    """
    This single test function will iterate through all metrics and all edge cases.
    """
    rm, dims, weights = rm_data
    # Get the `rm` object verb method based on its name (e.g., `rm.EVS`).
    metric_func = getattr(rm, metric_name)

    # Initialize dynamic keyword arguments
    kwargs = {}

    # Check if the current metric requires X_shape
    metrics_requiring_xshape = ["AR2", "ACOD", "RSE"]
    if metric_name in metrics_requiring_xshape:
        # Mocking a dataset with 3 features: (n_samples, n_features)
        kwargs["X_shape"] = (100, 7)

    if dims == 1:
        res = metric_func(**kwargs)
        assert isinstance(res, float), f"{metric_name} failed on 1D data. Expected float, got {type(res)}"

    elif dims == 2:     # Multi-output
        # 1. Test multi_output=None (Default)
        res_none = metric_func(multi_output=None, **kwargs)
        assert isinstance(res_none, float), f"{metric_name} with multi_output=None failed."

        # 2. Test multi_output="raw_values"
        res_raw = metric_func(multi_output="raw_values", **kwargs)
        assert isinstance(res_raw, (list, tuple, np.ndarray)), f"{metric_name} with multi_output='raw_values' failed."

        # 3. Test multi_output với weights custom
        res_weighted = metric_func(multi_output=weights, **kwargs)
        assert isinstance(res_weighted, float), f"{metric_name} with custom weights failed."


@pytest.mark.parametrize("metric_name", SAMPLE_WISE_METRICS)
def test_sample_wise_regression_metrics(rm_data, metric_name):
    """
    Test metrics that return element-wise or sample-wise errors.
    These metrics always return an iterable structure containing individual errors.
    """
    rm, dims, _ = rm_data
    metric_func = getattr(rm, metric_name)

    # Execute the sample-wise metric function
    res = metric_func()

    # For single scalar inputs, it might return a single numeric value depending on implementation.
    # Otherwise, it must return an array-like structure.
    if isinstance(rm.y_true, (int, float)):
        assert isinstance(res, (float, int, list, tuple, np.ndarray)), \
            f"{metric_name} failed on single scalar input."
    else:
        assert isinstance(res, (list, tuple, np.ndarray)), \
            f"{metric_name} failed. Expected an array-like container for per-sample outputs, got {type(res)}."
