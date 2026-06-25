RE - Relative Error
===================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Relative Error (RE)** is an element-wise metric used to evaluate the accuracy of a regression model by measuring the ratio of the absolute error to the magnitude of the actual (ground truth) value.

Like Absolute Error (AE), it computes the error for each individual data point. However, RE normalizes this error, making it a scale-independent ratio.

.. math::

    \text{RE}_i = \frac{|y_i - \hat{y}_i|}{|y_i|}

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Scale-independence:** Because it normalizes the error against the actual value, RE allows for fair comparison of prediction accuracy across different datasets or varying magnitudes within the same dataset.
	* **High interpretability:** It is extremely intuitive. When multiplied by 100, it directly translates into a percentage error (e.g., an RE of 0.15 means the prediction is off by 15% relative to the actual value).

**Disadvantages:**
	* **Zero-target vulnerability (Fatal flaw):** If the actual value (:math:`y_i`) is exactly zero, the formula attempts to divide by zero, resulting in an undefined value or program crash.
	* **Extreme volatility near zero:** If the actual value is very small (close to zero), even a negligible absolute deviation will result in a massively inflated Relative Error, distorting the model's perceived performance.
	* **Non-aggregated:** RE computes an array of individual errors. To evaluate the entire model at a macro level, it must be aggregated into metrics like Mean Absolute Percentage Error (MAPE).

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
    # Calculate Relative Error for each element
    print("RE: ", evaluator.RE())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate Relative Error for multi-dimensional array
    print("RE (Multi-output): ", evaluator.RE())
