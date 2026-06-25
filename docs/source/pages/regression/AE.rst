AE - Absolute Error
===================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

**Absolute Error (AE)** measures the magnitude of difference between an actual value and a predicted value for each individual data point, without considering the direction of the error.

Unlike aggregated metrics (such as MSE or RMSE), AE computes the element-wise error between a pair of lists, tuples, or NumPy arrays.

.. math::

    \text{AE}_i = | y_i - \hat{y}_i |

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Perfect interpretability:** Expressed in the exact same units as the response variable. It directly answers the question: "How far off was this specific prediction?"
	* **Outlier robustness:** Because it scales linearly (not squared), AE does not disproportionately inflate the error for extreme outliers.

**Disadvantages:**
	* **Non-aggregated:** AE is an element-wise metric. To evaluate overall model performance across an entire dataset, it must be mathematically aggregated into metrics like Mean Absolute Error (MAE) or Sum of Absolute Errors (SAE).
	* **Mathematical optimization:** The absolute value function is non-differentiable at exactly zero (:math:`y_i = \hat{y}_i`), which can make it less ideal as a direct loss function for gradient-based algorithms.

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
    # Calculate Absolute Error for each element
    print("AE: ", evaluator.AE())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate Absolute Error for multi-dimensional array
    print("AE (Multi-output): ", evaluator.AE())
