MASE - Mean Absolute Scaled Error
=================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Mean Absolute Scaled Error (MASE)** :cite:`hyndman2006another` is a highly robust metric for forecasting accuracy proposed by Rob J. Hyndman and Anne B. Koehler in 2006.

It scales the Mean Absolute Error (MAE) of your model by the MAE of a simple "naive" baseline model (a random walk that simply predicts the value from the previous season). This makes MAE completely scale-independent without the fatal flaws of percentage errors.

.. math::

    \text{MASE}(y, \hat{y}) = \frac{ \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i| }{ \frac{1}{N-m} \sum_{i=m+1}^{N} |y_i - y_{i-m}| }

Where:

* :math:`N` is the total number of samples.
* :math:`m` is the seasonal period (lag). Use :math:`m=1` for non-seasonal data, :math:`m=4` for quarterly data, :math:`m=12` for monthly data, etc.

-------------------------------------------------------------------------------

Description
-----------

**Key Insight: The MAPE Killer**
Before 2006, the industry standard for scale-independent evaluation was MAPE. However, MAPE crashes when data contains zeros and is heavily biased toward under-predictions. MASE solves this completely. Because MASE scales the error by the *historical variance* (the denominator) rather than the *actual value at a specific point*, it never divides by zero (unless all historical values are completely identical) and penalizes over/under-predictions perfectly symmetrically.

**Advantages:**

* **Clear Baseline Interpretation:** * ``MASE < 1.0``: Your model performs *better* than the naive random walk forecast.
    * ``MASE = 1.0``: Your model is exactly as accurate as just repeating the previous period's value.
    * ``MASE > 1.0``: Your model performs *worse* than the naive baseline.
* **Scale-Independence:** Can be safely used to compare the accuracy of models across completely different datasets, scales, and product lines.
* **Zero-Value Safe:** Perfectly handles time-series data with zero or negative values.

**Disadvantages:**
	* **Data requirement for denominator:** The denominator requires calculating the naive error. If the dataset is extremely short or the seasonality factor :math:`m` is too large relative to the dataset length, the denominator cannot be reliably computed.
	* **Complex formulation:** It is mathematically denser than MAE or RMSE, making it slightly harder to explain to non-technical business stakeholders compared to MAPE.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better).
* **Baseline score:** ``1.0`` (Equivalent to a naive forecast).
* **Range:** ``[0.0, +inf)``
* **Mathematical Reference:** `Wikipedia - MASE <https://en.wikipedia.org/wiki/Mean_absolute_scaled_error>`_

-------------------------------------------------------------------------------

Example Usage
-------------

*Note: In default implementation without specific seasonal period arguments, `permetrics` typically evaluates the non-seasonal formulation (m=1).*

.. code-block:: python
    :emphasize-lines: 10, 18

    from numpy import array
    from permetrics.regression import RegressionMetric

    ## 1. For 1-D array (Single-output)
    y_true = array([3, -0.5, 2, 7])
    y_pred = array([2.5, 0.0, 2, 8])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate Mean Absolute Scaled Error
    print("MASE: ", evaluator.MASE())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("MASE (Multi-output): ", evaluator.MASE(multi_output="raw_values"))
