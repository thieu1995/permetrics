CRM - Coefficient of Residual Mass
==================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Coefficient of Residual Mass (CRM)** :cite:`almodfer2022modeling` is a systemic bias metric widely used in environmental engineering and hydrology (e.g., water quality, sediment transport modeling).

It evaluates the tendency of a model to systematically overestimate or underestimate the dependent variable by comparing the sum of predicted values against the sum of actual (observed) values.

.. math::

    \text{CRM}(y, \hat{y}) = \frac{\sum_{i=1}^{N} y_i - \sum_{i=1}^{N} \hat{y}_i}{\sum_{i=1}^{N} y_i}

Note: Depending on the specific literature, the numerator is sometimes written as :math:`\sum \hat{y}_i - \sum y_i`. Both forms evaluate the same bias, just with inverted signs.

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Systemic Bias Detection:** CRM is an excellent diagnostic tool. A positive CRM indicates that the model systematically *underestimates* the actual values, while a negative CRM indicates systematic *overestimation*.
	* **Domain Specificity:** Highly interpretable in mass-balance systems (like fluid dynamics or sediment transport) where knowing the *total cumulative volume* error is more critical than individual step errors.

**Disadvantages:**
	* **The Cancellation Effect (Crucial Flaw):** CRM is **NOT** a measure of absolute accuracy. Because it simply sums the raw values, massive positive errors and massive negative errors can cancel each other out. A model could have terrible individual predictions but still achieve a perfect CRM of ``0.0``. CRM must *always* be used alongside metrics like RMSE or MAE.
	* **Zero-Sum Vulnerability:** If the sum of the actual values (:math:`\sum_{i=1}^{N} y_i`) equals exactly zero, the formula will divide by zero and crash (returning NaN/Undefined).

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Indicates no systemic bias; the total mass of predictions equals the total mass of observations).
* **Range:** ``(-inf, +inf)``
* **Mathematical Reference:** `ScienceDirect (CSITE) <https://doi.org/10.1016/j.csite.2022.101797>`_

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
    # Calculate Coefficient of Residual Mass
    print("CRM: ", evaluator.CRM())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("CRM (Multi-output): ", evaluator.CRM(multi_output="raw_values"))
