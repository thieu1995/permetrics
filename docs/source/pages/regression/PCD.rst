PCD - Prediction of Change in Direction
=======================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Prediction of Change in Direction (PCD)**, often referred to as **Directional Accuracy**, evaluates a regression model's ability to correctly forecast the *trend* (upward or downward movement) of a time series, regardless of the predicted magnitude.

.. math::

    \text{PCD}(y, \hat{y}) = \frac{1}{N-1} \sum_{i=2}^{N} I\left( (\hat{y}_i - \hat{y}_{i-1})(y_i - y_{i-1}) > 0 \right)

Note: :math:`\hat{y}_i` and :math:`y_i` are the predicted and actual values at time :math:`i`, respectively. :math:`N` is the total number of observations, and :math:`I(\cdot)` is the indicator function which equals ``1`` if the condition is true and ``0`` otherwise.

-------------------------------------------------------------------------------

Description
-----------

**Key Insight: The Algorithmic Trading Standard**
In financial forecasting, stock market prediction, and macroeconomic trend analysis, PCD is arguably more critical than magnitude-based metrics like RMSE or MAE. In trading, correctly predicting that a price will go *up* (even if you underestimate by how much) leads to profit. Conversely, a model with a tiny RMSE that consistently predicts a slight increase when the actual asset slightly decreases will lead to catastrophic financial losses. PCD directly measures profitability potential.

**Advantages:**
	* **Magnitude Agnostic:** It completely isolates the model's phase-tracking ability from its volume-tracking ability. It doesn't care if the prediction is off by 1 unit or 1000 units, as long as the direction of change from the previous step is correct.
	* **Trend Diagnostic:** Highly effective for evaluating time-series models (like ARIMA, LSTM) to ensure they aren't just naively predicting a flat line or repeating the previous day's value.

**Disadvantages:**
	* **Data Length Trap (Critical Flaw):** Because the formula divides by :math:`N-1`, the metric requires **at least 2 data points** to compute a direction. If an array of length 1 is passed, the calculation will trigger a division-by-zero error. *(Implementation note: Ensure your code raises a proper `ValueError` if* :math:`N < 2` *).*
	* **Flatline Ambiguity:** If the actual value does not change from the previous step (:math:`y_i - y_{i-1} = 0`), the product becomes zero. The strict inequality (:math:`> 0`) means the model is implicitly penalized (scores 0) even if it perfectly predicted the flatline.
	* **Ignores Severity:** Predicting a 1% drop when the market crashes by 50% yields a perfect PCD score for that step, dangerously masking the severity of the real-world event.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (100% of the directional changes were correctly predicted).
* **Baseline score:** ``0.5`` (Equivalent to a random coin flip for direction).
* **Range:** ``[0.0, 1.0]``

-------------------------------------------------------------------------------

Example Usage
-------------

*Note: The input arrays must contain at least two sequential elements (N >= 2).*

.. code-block:: python
    :emphasize-lines: 10, 18

    from numpy import array
    from permetrics.regression import RegressionMetric

    ## 1. For 1-D array (Single-output)
    y_true = array([3, -0.5, 2, 7])
    y_pred = array([2.5, 0.0, 2, 8])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate Prediction of Change in Direction
    print("PCD: ", evaluator.PCD())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("PCD (Multi-output): ", evaluator.PCD(multi_output="raw_values"))
