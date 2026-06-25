AR - Absolute Pearson's Correlation Coefficient
===============================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Absolute Pearson's Correlation Coefficient (AR)** is a custom evaluation metric introduced natively in ``permetrics``. It modifies the standard Pearson's Correlation Coefficient by taking the absolute value of the deviations from the mean in the numerator.

.. math::

    \text{AR}(y, \hat{y}) = \frac{\sum_{i=1}^{N} |y_i - \bar{y}| |\hat{y}_i - \bar{\hat{y}}|}{\sqrt{\sum_{i=1}^{N} (y_i - \bar{y})^2} \sqrt{\sum_{i=1}^{N} (\hat{y}_i - \bar{\hat{y}})^2}}

Note: :math:`\bar{y}` and :math:`\bar{\hat{y}}` represent the mean of the actual and predicted values, respectively.

-------------------------------------------------------------------------------

Description
-----------

Because AR enforces absolute values in the covariance-like numerator, it behaves very differently from standard correlation:

**Advantages:**
	* **Volatility and Magnitude Tracking:** AR strictly measures whether the *magnitude* of variation in your predictions matches the true data, regardless of phase. This makes it a highly specialized metric for evaluating volatility models (e.g., financial market fluctuations, seismic signal processing) where predicting the *size* of a movement is the primary goal, even if the exact direction is wrong.

**Disadvantages:**
	* **Directional Blindness (Crucial Flaw):** AR completely ignores the phase or direction of the relationship. If your actual data spikes by +10 units, but your model predicts a drop of -10 units, standard Pearson R would penalize it severely. However, AR will score it perfectly (``1.0``) because the absolute magnitude of the deviation is identical. It should **never** be used as a standalone accuracy metric for standard regression tasks.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Bigger value is better).
* **Range:** ``[0.0, 1.0]``

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
    # Calculate Absolute Pearson's Correlation Coefficient
    print("AR: ", evaluator.absolute_pearson_correlation_coefficient())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("AR (Multi-output): ", evaluator.AR(multi_output="raw_values"))
