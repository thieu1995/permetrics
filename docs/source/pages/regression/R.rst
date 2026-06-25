R - Pearson's Correlation Coefficient
=====================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

**Pearson’s Correlation Coefficient** (often denoted as **R** or **PCC**) :cite:`van2023groundwater` is a foundational statistical measure that quantifies the strength and direction of the linear relationship between two variables (the actual values and the predicted values).

.. math::

    \text{R}(y, \hat{y}) = \frac{\sum_{i=1}^{N} (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum_{i=1}^{N} (y_i - \bar{y})^2} \sqrt{\sum_{i=1}^{N} (\hat{y}_i - \bar{\hat{y}})^2}}

Note: :math:`\bar{y}` and :math:`\bar{\hat{y}}` represent the mean of the actual and predicted values, respectively.

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Trend identification:** Excellent at evaluating whether the model correctly captures the directional trend of the data. A value close to ``+1`` indicates a strong positive correlation, while a value close to ``-1`` indicates a strong negative correlation.
	* **Scale-invariant:** Because it normalizes the covariance by the standard deviations, R is dimensionless. You can seamlessly compare the linear correlation of models trained on entirely different datasets.

**Disadvantages:**
	* **Linear constraint:** The most critical limitation is that R *only* evaluates linear relationships. A model might capture a perfect, highly complex non-linear relationship (e.g., sinusoidal or quadratic) but still score an R near ``0.0``.
	* **Ignores absolute error:** R measures linear tracking, not absolute accuracy. If actual values are ``[1, 2, 3]`` and predictions are ``[10, 20, 30]``, the R score will be a perfect ``1.0``, completely ignoring the massive magnitude gap. It must be paired with error metrics like RMSE or MAE.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Perfect positive linear relationship) or ``-1.0`` (Perfect negative linear relationship). A value of ``0.0`` indicates no linear relationship.
* **Range:** ``[-1.0, 1.0]``

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
    # Calculate Pearson's Correlation Coefficient
    print("R: ", evaluator.pearson_correlation_coefficient())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("R (Multi-output): ", evaluator.R(multi_output="raw_values"))
