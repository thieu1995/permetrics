WI - Willmott Index of Agreement
================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Willmott Index** :cite:`da2017reference`, widely known in scientific literature as the **Index of Agreement (d)**, was developed by Cort J. Willmott (1981) to overcome the insensitivity of correlation-based measures to differences in the observed and predicted means and variances.

It represents the ratio of the mean square error to the "potential error," providing a standardized measure of the degree of model prediction error.

.. math::

    \text{WI}(y, \hat{y}) = 1 - \frac{ \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 }{ \sum_{i=1}^{N} \left( |\hat{y}_i - \bar{y}| + |y_i - \bar{y}| \right)^2 }

Note: :math:`\bar{y}` represents the mean of the actual observed values. The denominator represents the maximum possible sum of squared errors.

-------------------------------------------------------------------------------

Description
-----------

**Key Insight: WI vs. Pearson Correlation (R)**
The Pearson Correlation (R) can be misleadingly high even if a model's predictions are systematically biased (e.g., if the model always predicts exactly double the true value, R will still be 1.0). Willmott's Index of Agreement explicitly solves this by penalizing additive and proportional differences in the observed and simulated means and variances. It strictly measures *absolute agreement*, not just linear correlation.

**Advantages:**
	* **Strict Bounding:** Unlike NSE or R2, which can approach negative infinity, WI is strictly bounded between ``0.0`` and ``1.0``. This makes it extremely stable for cross-model comparisons and multi-site averaging without the risk of a single catastrophic model skewing the mean.
	* **Hydrological Standard:** It is a mandatory evaluation metric in many high-impact climate, evapotranspiration, and hydrology journals.

**Disadvantages:**
	* **Outlier Sensitivity:** Because both the numerator and denominator square the errors, the standard WI is highly sensitive to extreme outliers. (Willmott later proposed a "modified index of agreement" using absolute values to address this, but the squared version remains the most widely cited).
	* **High-Value Bias:** WI tends to yield relatively high values (e.g., > 0.6) even for poor models, meaning the visual interpretation of a "good" score must be strictly calibrated (often requiring scores > 0.85 to be considered acceptable).

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates perfect agreement).
* **Worst possible score:** ``0.0`` (Indicates complete disagreement).
* **Range:** ``[0.0, 1.0]``
* **Mathematical Reference:** `Reference evapotranspiration estimation methods <https://www.researchgate.net/publication/319699360_Reference_evapotranspiration_for_Londrina_Parana_Brazil_performance_of_different_estimation_methods>`_

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
    # Calculate Willmott Index of Agreement
    print("WI: ", evaluator.WI())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("WI (Multi-output): ", evaluator.WI(multi_output="raw_values"))
