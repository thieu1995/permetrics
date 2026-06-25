RRSE - Root Relative Squared Error
==================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Root Relative Squared Error (RRSE)** is a regression metric that evaluates the performance of a predictive model relative to a simple baseline model (which continuously predicts the mean of the actual values).

Mathematically, it is the square root of the Relative Squared Error (RSE). By taking the square root, the error scales linearly with the magnitude of the predictions, making it directly comparable to the Root Mean Square Error (RMSE) but normalized as a dimensionless ratio.

.. math::

    \text{RRSE}(y, \hat{y}) = \sqrt{ \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2} }

Note: :math:`\bar{y}` represents the mean of the actual observed values.

-------------------------------------------------------------------------------

Description
-----------

**Key Insight:**
    * **RRSE vs. RAE:** While Relative Absolute Error (RAE) uses absolute differences (L1 Norm) to treat all errors linearly, RRSE uses squared differences (L2 Norm). This mathematical distinction means RRSE heavily penalizes large individual errors, whereas RAE is more robust to extreme outliers.

**Advantages:** Like RAE, it provides a strictly defined benchmark:
    * ``RRSE < 1.0``: The model outperforms the simple mean-predicting baseline.
    * ``RRSE = 1.0``: The model is only as accurate as predicting the baseline mean.
    * ``RRSE > 1.0``: The model performs *worse* than the simple baseline.
* **Scale-independent:** Because the squared error is divided by the data's inherent variance, the metric is dimensionless. This allows for fair comparisons of model accuracy across entirely different domains and datasets.
* **Large Error Identification:** By squaring the residuals before summing, RRSE aggressively highlights models that occasionally make massive prediction errors.

**Disadvantages:**
    * **Extreme Outlier Sensitivity:** The L2 Norm nature of this metric makes it highly vulnerable to single massive outliers, which can disproportionately inflate the RRSE score and mask the model's performance on the rest of the dataset.
    * **Zero-Variance Vulnerability:** If the ground truth dataset contains identical values (zero variance, meaning :math:`y_i = \bar{y}` for all :math:`i`), the denominator becomes exactly zero, causing the calculation to crash or return an undefined/NaN value.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better, indicating zero residual error).
* **Range:** ``[0.0, +inf)``
* **Mathematical Reference:** `WEKA Machine Learning Evaluation <https://waikato.github.io/weka-wiki/formats_and_processing/evaluating_models/>`_

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
    # Calculate Root Relative Squared Error
    print("RRSE: ", evaluator.RRSE())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("RRSE (Multi-output): ", evaluator.RRSE(multi_output="raw_values"))
