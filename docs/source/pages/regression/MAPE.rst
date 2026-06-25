MAPE - Mean Absolute Percentage Error
=====================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Mean Absolute Percentage Error (MAPE)** :cite:`nguyen2020new` is a widely used statistical measure of prediction accuracy in forecasting, business, and economics.

Mathematically, it is the Mean Relative Error (MRE) multiplied by 100 to express the error as a percentage. It measures the average magnitude of the error relative to the actual value.

.. math::

    \text{MAPE}(y, \hat{y}) = \frac{100\%}{N} \sum_{i=1}^{N} \frac{|y_i - \hat{y}_i|}{|y_i|}

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Ultimate Interpretability:** MAPE is arguably the most intuitive metric for non-technical stakeholders (e.g., management, clients). Stating "the model has a 5% error rate" is universally understood, whereas stating "the model has an RMSE of 12.4" requires deep domain context.
	* **Scale-Independence:** Because it is a percentage, you can use MAPE to compare the performance of models across different datasets, varying scales, or different product lines (e.g., predicting sales for a $10 item vs. a $1000 item).

**Disadvantages:**
	* **The Zero-Value Trap (Critical Flaw):** Just like MRE, if the actual ground truth value (:math:`y_i`) is exactly ``0.0``, the calculation involves division by zero and will crash or return an infinite value. It cannot be used directly on intermittent demand datasets with zero-sales days.
	* **Asymmetric Penalization (Hidden Bias):** MAPE implicitly favors models that under-predict rather than over-predict. For example, if the actual value is 100 and the model predicts 50 (under-prediction by 50), the MAPE is 50%. However, if the model predicts 150 (over-prediction by 50), the MAPE is also 50%. But consider extreme cases: an under-prediction is bounded (you can't predict less than 0, so max error is 100%), while an over-prediction has no upper bound (predicting 400 for a target of 100 yields a 300% error). This often forces models trained on MAPE to systematically under-forecast.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Indicates perfect forecast accuracy).
* **Range:** ``[0.0, +inf)``
* **Mathematical Reference:** `Institute of Business Forecasting <https://ibf.org/knowledge/glossary/mape-mean-absolute-percentage-error-174>`_

-------------------------------------------------------------------------------

Example Usage
-------------

*Note: Ensure your ground truth data does not contain zero values to avoid division-by-zero errors. Data should ideally be strictly positive.*

.. code-block:: python
    :emphasize-lines: 10, 18

    from numpy import array
    from permetrics.regression import RegressionMetric

    ## 1. For 1-D array (Single-output)
    y_true = array([3, 0.5, 2, 7])
    y_pred = array([2.5, 0.6, 2, 8])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate Mean Absolute Percentage Error
    print("MAPE: ", evaluator.MAPE())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [0.1, 1], [7, 6]])
    y_pred = array([[0.6, 2], [0.1, 2], [8, 5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("MAPE (Multi-output): ", evaluator.MAPE(multi_output="raw_values"))
