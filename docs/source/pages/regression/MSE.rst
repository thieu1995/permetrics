MSE - Mean Squared Error
========================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Mean Squared Error (MSE)** is a fundamental risk metric that measures the expected value of the squared (quadratic) error or loss. It calculates the average of the squares of the errors—that is, the average squared difference between the predicted values (:math:`\hat{y}`) and the actual values (:math:`y`).

.. math::

    \text{MSE}(y, \hat{y}) = \frac{1}{N} \sum_{i=0}^{N - 1} (y_i - \hat{y}_i)^2

-------------------------------------------------------------------------------

Description
-----------

**Advantages**:
	* **Mathematical tractability:** MSE is strictly convex and smooth (differentiable). This makes it the gold standard loss function for optimization algorithms (e.g., Gradient Descent) in machine learning and deep learning.
	* **Outlier penalization:** By squaring the errors, MSE heavily penalizes larger errors. This is highly beneficial when large prediction deviations are exponentially more costly than small ones.

**Disadvantages**:
	* **Lack of interpretability:** The final score is in *squared units* of the target variable, making it unintuitive to communicate business/clinical impact directly (unlike MAE or RMSE).
	* **Extreme sensitivity to noisy outliers:** If the dataset contains severe outliers that are merely data errors (noise) rather than true signals, MSE will distort the overall model evaluation significantly.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better).
* **Range:** ``[0, +inf)``
* **Mathematical Reference:** `Statistics By Jim <https://statisticsbyjim.com/regression/mean-squared-error-mse/>`_

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
    # Calculate MSE
    print("MSE: ", evaluator.MSE())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("MSE (Multi-output): ", evaluator.MSE(multi_output="raw_values"))
