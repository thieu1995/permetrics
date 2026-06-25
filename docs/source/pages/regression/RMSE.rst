RMSE - Root Mean Square Error
=============================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

**Root Mean Square Error (RMSE)** :cite:`nguyen2019building` is a standard statistical metric used to evaluate the accuracy of regression and time series models.
It measures the square root of the average squared differences between predicted values (:math:`\hat{y}`) and actual values (:math:`y`).

.. math::

    \text{RMSE}(y, \hat{y}) = \sqrt{\frac{1}{N} \sum_{i=0}^{N - 1} (y_i - \hat{y}_i)^2}

-------------------------------------------------------------------------------

Description
-----------

RMSE represents the sample standard deviation of the differences between predicted values and observed values (residuals).

* **Sensitivity:** Highly sensitive to large errors (outliers) because the errors are squared before being averaged.
* **Scale-dependency:** The output is in the same units as the response variable. This makes interpretation intuitive within a single dataset, but **unsuitable for comparing models across datasets with different scales** (use NRMSE instead).

* The RMSE is a widely used measure of forecast accuracy because it is sensitive to both the magnitude and direction of the errors. A lower RMSE indicates better forecast accuracy. However, it has a drawback that it is not normalized, meaning that it is dependent on the scale of the response variable. Therefore, it is difficult to compare the RMSE values across different datasets with different scales.

* The RMSE is commonly used in various fields, including finance, economics, and engineering, to evaluate the performance of forecasting models. It is often used in conjunction with other measures, such as the Mean Absolute Error (MAE) and the Mean Absolute Percentage Error (MAPE), to provide a more comprehensive evaluation of the model's performance.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value indicates better forecast accuracy).
* **Range:** ``[0, +inf)``

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 10, 18

    from numpy import array
    from permetrics.regression import RegressionMetric

    ## 1. For 1-D array (Single-output)
    y_true = array([3, -0.5, 2, 7])
    y_pred = array([2.5, 0.0, 2, 8])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate RMSE
    print("RMSE: ", evaluator.RMSE())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("RMSE (Multi-output): ", evaluator.RMSE(multi_output="raw_values"))
