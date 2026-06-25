COR - Correlation Coefficient
=============================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

**Correlation (COR)** :cite:`joreskog1978structural` (specifically Pearson's Correlation Coefficient) measures the strength and direction of the linear relationship between the actual values (:math:`y`) and predicted values (:math:`\hat{y}`).

It is calculated by dividing the covariance of the variables by the product of their standard deviations. This normalization scales the output to a specific range, making it dimensionless and entirely independent of the units of measurement.

.. math::

    \text{COR}(y, \hat{y}) = \frac{\text{cov}(y, \hat{y})}{\sigma_y \sigma_{\hat{y}}} = \frac{\sum_{i=1}^{N} (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum_{i=1}^{N} (y_i - \bar{y})^2 \sum_{i=1}^{N} (\hat{y}_i - \bar{\hat{y}})^2}}

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Scale-invariant:** Because it is dimensionless, you can use COR to compare the predictive behavior of models across entirely different datasets and domains.
	* **Trend evaluation:** Highly effective at identifying whether the model correctly captures the directional trend of the data (e.g., when the actual value goes up, the prediction also goes up).

**Disadvantages:**
	* **Ignores magnitude (Crucial limitation):** COR measures linear tracking, *not* absolute accuracy. If actual values are `[1, 2, 3]` and predictions are `[1000, 2000, 3000]`, the COR will be a perfect `1.0`, even though the absolute errors are massive. It should always be paired with an error metric like RMSE or MAE.
	* **Linearity assumption:** It only evaluates linear relationships. A model might capture a perfect non-linear relationship (e.g., quadratic) but still score a low or zero Pearson correlation.
	* **Outlier sensitivity:** A single massive outlier can heavily distort the covariance, drastically shifting the correlation score (as demonstrated in Anscombe's quartet).

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Perfect positive correlation). A score of ``-1.0`` indicates perfect negative correlation, and ``0.0`` indicates no linear correlation.
* **Range:** ``[-1.0, 1.0]``
* **Mathematical Reference:** `Corporate Finance Institute <https://corporatefinanceinstitute.com/resources/data-science/covariance/>`_

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
    # Calculate Correlation Coefficient
    print("COR: ", evaluator.COR())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("COR (Multi-output): ", evaluator.COR(multi_output="raw_values"))
