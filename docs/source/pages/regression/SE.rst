SE - Squared Error
==================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

**Squared Error (SE)** is an element-wise metric that measures the square of the difference between an actual value and a predicted value for each individual data point.

Unlike Mean Squared Error (MSE), which computes the average over an entire dataset, SE computes an array of squared errors for a pair of lists, tuples, or NumPy arrays.

.. math::

    \text{SE}_i = (y_i - \hat{y}_i)^2

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Mathematical tractability:** The squaring operation makes the function strictly convex and differentiable everywhere. This is highly advantageous for calculating gradients in optimization algorithms.
	* **Aggressive outlier penalization:** By squaring the error, SE exponentially magnifies larger deviations. This makes it extremely useful when you want to isolate and identify individual predictions that failed massively.

**Disadvantages:**
	* **Lack of interpretability:** The error is expressed in *squared units* of the target variable, making it completely unintuitive for clinical or business communication.
	* **Non-aggregated:** SE is an element-wise array of scores. To evaluate the overall predictive performance across an entire model, it must be aggregated into MSE, RMSE, or Sum of Squared Errors (SSE).

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better).
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
    # Calculate Squared Error for each element
    print("SE: ", evaluator.SE())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate Squared Error for multi-dimensional array
    print("SE (Multi-output): ", evaluator.SE())
