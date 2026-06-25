DRV - Deviation of Runoff Volume
================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Deviation of Runoff Volume (DRV)** is a domain-specific metric primarily utilized in hydrology and environmental engineering.

It evaluates the overall mass balance of a model by calculating the ratio of the total observed (actual) runoff volume to the total simulated (predicted) runoff volume over a specified period.

.. math::

    \text{DRV}(y, \hat{y}) = \frac{\sum_{i=1}^{N} y_i}{\sum_{i=1}^{N} \hat{y}_i}

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Mass-balance verification:** DRV is exceptionally useful for verifying if a model correctly predicts the *total volume* of an event (e.g., total rainfall, total flood volume), even if the exact timing of the predictions is slightly off.
	* **Highly interpretable:** A score of ``1.0`` means perfect volume matching. A score of ``0.5`` implies the model over-predicted the total volume by a factor of 2. A score of ``2.0`` means it under-predicted by half.

**Disadvantages:**
	* **The Cancellation Effect (Crucial Flaw):** Like the Coefficient of Residual Mass (CRM), DRV aggregates all values before comparing. Massive under-predictions on day 1 can perfectly cancel out massive over-predictions on day 2, yielding a "perfect" DRV of 1.0 despite terrible daily accuracy. It must be paired with time-step metrics like RMSE or NSE.
	* **Zero-Sum Vulnerability:** If the sum of the predicted values (:math:`\sum_{i=1}^{N} \hat{y}_i`) is exactly zero (e.g., a model predicts absolutely zero runoff for the entire period), the metric will crash due to division by zero.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates perfect agreement between total actual and total predicted volumes. **Closer to 1.0 is better**).
* **Range:** ``(-inf, +inf)`` (Practically ``[0, +inf)`` since physical volumes are typically non-negative).
* **Mathematical Reference:** `RStudio Pubs <https://rstudio-pubs-static.s3.amazonaws.com/433152_56d00c1e29724829bad5fc4fd8c8ebff.html>`_

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
    # Calculate Deviation of Runoff Volume
    print("DRV: ", evaluator.DRV())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("DRV (Multi-output): ", evaluator.DRV(multi_output="raw_values"))
