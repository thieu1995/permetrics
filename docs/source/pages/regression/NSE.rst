NSE - Nash-Sutcliffe Efficiency
===============================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Nash-Sutcliffe Efficiency (NSE)** :cite:`xie2021predicting` is a normalized statistic that determines the relative magnitude of the residual variance compared to the measured data variance.

While mathematically identical to the Coefficient of Determination (:math:`R2`) and the Efficiency Coefficient (EC), the term NSE is strictly the industry standard in hydrology and earth sciences for assessing the predictive skill of hydrological models (e.g., simulating streamflow or watershed discharge).

.. math::

    \text{NSE}(y, \hat{y}) = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}

Note: :math:`\bar{y}` represents the mean of the actual observed values.

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**

* **Standardized Benchmark:** Like :math:`R2`, it provides a highly intuitive evaluation scale for time-series forecasting:
    * ``NSE = 1.0``: Perfect match between modeled and observed data.
    * ``NSE = 0.0``: The model predictions are exactly as accurate as the mean of the observed data.
    * ``NSE < 0.0``: The observed mean is a better predictor than the model.
* **Domain Specificity:** Using NSE over R2 in environmental engineering publications signals domain expertise and adheres to the standard reporting protocols of hydrological journals.

**Disadvantages:**
	* **Outlier Sensitivity:** Because the differences are squared in both the numerator and denominator, NSE is highly sensitive to extreme values (e.g., massive but brief flood peaks). A model might perform excellently during low-flow periods but receive a poor NSE score if it misses a single extreme high-flow event.
	* **Mathematical Redundancy:** Outside of hydrology and environmental sciences, calculating NSE provides no additional mathematical value over the standard R2 metric.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Bigger value is better).
* **Range:** ``(-inf, 1.0]``
* **Mathematical Reference:** `Agrimetsoft NSE Calculator <https://agrimetsoft.com/calculators/Nash%20Sutcliffe%20model%20Efficiency%20coefficient>`_

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
    # Calculate Nash-Sutcliffe Efficiency
    print("NSE: ", evaluator.NSE())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("NSE (Multi-output): ", evaluator.NSE(multi_output="raw_values"))
