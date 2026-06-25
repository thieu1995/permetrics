KGE - Kling-Gupta Efficiency
============================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Kling-Gupta Efficiency (KGE)** :cite:`van2023groundwater` is a robust statistical metric developed primarily for hydrology to overcome the limitations of the Nash-Sutcliffe Efficiency (NSE) and Mean Squared Error (MSE).

While NSE and MSE implicitly bundle different types of errors together, KGE explicitly decomposes model performance into three distinct mathematical components: correlation (timing), bias (volume), and variability (amplitude).

.. math::

    \text{KGE}(y, \hat{y}) = 1 - \sqrt{(r - 1)^2 + (\beta - 1)^2 + (\gamma - 1)^2}

Where the three components are:

1. **Correlation (** :math:`r` **):** The Pearson correlation coefficient between actual and predicted values.
2. **Bias Ratio (** :math:`\beta` **):** The ratio of the predicted mean to the actual mean (:math:`\mu_{\hat{y}} / \mu_y`).
3. **Variability Ratio (** :math:`\gamma` **):** The ratio of the coefficients of variation (:math:`\text{CV}_{\hat{y}} / \text{CV}_y`), which mathematically simplifies to :math:`(\sigma_{\hat{y}} / \mu_{\hat{y}}) / (\sigma_y / \mu_y)`.

Note: Some literature defines KGE using the ratio of standard deviations :math:`\sigma_{\hat{y}} / \sigma_y` instead of CV. ``permetrics`` implements the revised KGE' which uses the CV ratio for better theoretical properties.

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Diagnostic Power:** By breaking the error down into three components (Euclidean distance in a 3D space), KGE allows researchers to diagnose *exactly why* a model is failing. Is it failing to capture the trend (:math:`r`), consistently underestimating volume (:math:`\beta`), or failing to capture the extreme peaks and lows (:math:`\gamma`)?
	* **Solves NSE Variance Bias:** NSE tends to implicitly penalize overestimation of variance more than underestimation, leading models calibrated on NSE to produce overly smooth, "flat" predictions. KGE fixes this by explicitly rewarding models that match the true variance.

**Disadvantages:**
	* **The Zero-Mean Trap (Critical Flaw):** Because the formulas for :math:`\beta` and :math:`\gamma` require dividing by the actual mean (:math:`\mu_y`), KGE is **strictly designed for strictly positive variables** (like streamflow, rainfall, or sales). If applied to standardized data (mean = 0) or data crossing zero (like Celsius temperature), the division by zero will cause the metric to crash or return infinite values.
	* **Lack of a clear baseline:** Unlike NSE where ``0.0`` explicitly means "as good as predicting the mean," the mean-prediction baseline for KGE is approximately ``-0.41`` (specifically :math:`1 - \sqrt{2}`). Positive KGE scores are good, but interpreting scores between -0.41 and 0 requires caution.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates perfect correlation, perfect bias match, and perfect variability match).
* **Range:** ``(-inf, 1.0]``
* **Mathematical Reference:** `RStudio Pubs - KGE <https://rstudio-pubs-static.s3.amazonaws.com/433152_56d00c1e29724829bad5fc4fd8c8ebff.html>`_

-------------------------------------------------------------------------------

Example Usage
-------------

*Note: Ensure that the mean of your ground truth data is not zero or extremely close to zero to avoid mathematical instability.*

.. code-block:: python
    :emphasize-lines: 10, 18

    from numpy import array
    from permetrics.regression import RegressionMetric

    ## 1. For 1-D array (Single-output)
    y_true = array([3, 0.5, 2, 7])
    y_pred = array([2.5, 0.0, 2, 8])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate Kling-Gupta Efficiency
    print("KGE: ", evaluator.KGE())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [1, 1], [7, 6]])
    y_pred = array([[0, 2], [1, 2], [8, 5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("KGE (Multi-output): ", evaluator.KGE(multi_output="raw_values"))
