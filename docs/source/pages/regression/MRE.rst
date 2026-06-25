MRE - Mean Relative Error
=========================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Mean Relative Error (MRE)** (also referred to as **Mean Relative Bias**) measures the average of the relative errors between the actual and predicted values. It represents the error as a percentage or ratio of the true value, rather than an absolute difference.

.. math::

    \text{MRE}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} \frac{|y_i - \hat{y}_i|}{|y_i|}

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Scale-invariant Comparison:** Unlike MAE or RMSE, MRE expresses error as a percentage/ratio of the ground truth. This is invaluable when comparing the predictive accuracy of models across datasets with entirely different units (e.g., comparing a model predicting milligrams vs. a model predicting kilograms).
	* **Interpretability:** An MRE of ``0.10`` immediately tells the user that the model is off by approximately 10% on average, which is often more intuitive for business stakeholders than an absolute error of ``500 units``.

**Disadvantages:**
	* **The Zero-Value Trap (Critical Flaw):** If any single value in the ground truth (:math:`y_i`) is exactly ``0.0``, the division by zero will cause the calculation to crash or return ``Inf/NaN``.
	* **Asymmetric Penalization:** Because the true value is in the denominator, MRE is much more sensitive to errors on small target values than on large ones. A prediction error of ``1`` on a target of ``10`` results in an error of ``0.1``, but an error of ``1`` on a target of ``1000`` results in a negligible error of ``0.001``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better).
* **Range:** ``[0.0, +inf)``

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 10, 18

    from numpy import array
    from permetrics.regression import RegressionMetric

    ## 1. For 1-D array (Single-output)
    y_true = array([3, 0.5, 2, 7])
    y_pred = array([2.5, 0.6, 2, 8])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate Mean Relative Error
    print("MRE: ", evaluator.MRE())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [0.1, 1], [7, 6]])
    y_pred = array([[0.6, 2], [0.1, 2], [8, 5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("MRE (Multi-output): ", evaluator.MRE(multi_output="raw_values"))
