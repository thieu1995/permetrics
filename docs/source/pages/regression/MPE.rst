MPE - Mean Percentage Error
===========================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Mean Percentage Error (MPE)** is a statistical metric used to measure the systematic bias of a forecasting model in percentage terms.

Unlike the Mean Absolute Percentage Error (MAPE), MPE does not use absolute values. By preserving the sign of the errors, it allows positive and negative errors to cancel each other out, revealing whether the model has a consistent tendency to over-forecast or under-forecast.

.. math::

    \text{MPE}(y, \hat{y}) = \frac{100\%}{N} \sum_{i=1}^{N} \frac{y_i - \hat{y}_i}{y_i}

Note: In ``permetrics``, the numerator is calculated as Actual minus Predicted :math:`(y_i - \hat{y}_i)`. Therefore, a positive MPE indicates that the model systematically under-predicts, while a negative MPE indicates over-prediction.

-------------------------------------------------------------------------------

Description
-----------

**Key Insight: The Percentage Bias Indicator**
MPE is to MAPE exactly what Mean Bias Error (MBE) is to Mean Absolute Error (MAE). Because errors cancel out, a perfect MPE of ``0.0`` does **not** mean the model's predictions are perfectly accurate; it simply means that the percentage of over-predictions perfectly balances the percentage of under-predictions. MPE should **never** be used alone to evaluate accuracy; it is a diagnostic tool meant to be paired with MAPE or RMSE.

**Advantages:**
	* **Directional Percentage Diagnostic:** It provides an incredibly intuitive way for businesses to understand systematic bias. Telling stakeholders "our inventory model has a -15% MPE" immediately communicates that you are systematically over-stocking by about 15%.
	* **Scale-Independence:** Because the error is normalized by the actual value, it can evaluate bias across entirely different datasets, product lines, or scales.

**Disadvantages:**
	* **The Zero-Value Trap (Critical Flaw):** Like all percentage errors, if the actual ground truth value (:math:`y_i`) is exactly ``0.0``, the calculation involves division by zero and will instantly crash or return ``NaN/Inf``.
	* **Cancellation Illusion:** Highly inaccurate models can still achieve an MPE close to zero if their wild over-predictions and under-predictions happen to average out.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Indicates zero systematic percentage bias).
* **Range:** ``(-inf, +inf)``
    * **MPE > 0:** The model systematically underestimates (Actual > Predicted).
    * **MPE < 0:** The model systematically overestimates (Actual < Predicted).
* **Mathematical Reference:** `Dataquest Regression Metrics <https://www.dataquest.io/blog/understanding-regression-error-metrics/>`_

-------------------------------------------------------------------------------

Example Usage
-------------

*Note: Ensure your ground truth data does not contain zero values to avoid division-by-zero errors. Data should ideally be strictly positive.*

.. code-block:: python
    :emphasize-lines: 10, 18

    from numpy import array
    from permetrics.regression import RegressionMetric

    ## 1. For 1-D array (Single-output)
    y_true = array([3, 0.5, 2, 7])
    y_pred = array([2.5, 0.6, 2, 8])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate Mean Percentage Error
    print("MPE: ", evaluator.MPE())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [0.1, 1], [7, 6]])
    y_pred = array([[0.6, 2], [0.1, 2], [8, 5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("MPE (Multi-output): ", evaluator.MPE(multi_output="raw_values"))
