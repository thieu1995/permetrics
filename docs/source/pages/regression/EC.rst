EC - Efficiency Coefficient
===========================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Efficiency Coefficient (EC)** :cite:`joreskog1978structural` (mathematically identical to the Nash-Sutcliffe Efficiency or the Coefficient of Determination :math:`R2`) is a statistical metric used to evaluate the predictive accuracy of continuous regression models.

It assesses the model's predictive skill relative to a baseline "no-knowledge" benchmark model (which simply predicts the mean of the observed data).

.. math::

    \text{EC}(y, \hat{y}) = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}

Note: :math:`\bar{y}` represents the mean of the actual observed values.

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**

* **Built-in Baseline Comparison:** Highly interpretable.
    * ``EC = 1.0``: Perfect prediction.
    * ``EC = 0.0``: The model is only as accurate as predicting the constant mean of the observed data.
    * ``EC < 0.0``: The model is *worse* than simply guessing the mean.
* **Scale-independent:** Because the error variance is normalized by the data's inherent variance, EC can be used to compare model performance across entirely different datasets and domains.

**Disadvantages:**
    * **Extreme Outlier Sensitivity:** Because both the numerator (residual error) and denominator (total variance) are squared, a single massive outlier can disproportionately crash the EC score, even if the model performs perfectly on 99% of the remaining data.
    * **Non-linear scaling:** The difference in model quality between an EC of 0.90 and 0.95 is vastly more significant than the difference between 0.20 and 0.25, making linear performance comparisons tricky.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Bigger value is better).
* **Range:** ``(-inf, 1.0]``
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
    # Calculate Efficiency Coefficient
    print("EC: ", evaluator.EC())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("EC (Multi-output): ", evaluator.EC(multi_output="raw_values"))
