CI - Confidence Index
=====================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Confidence Index (CI)** :cite:`ahmed2021comprehensive`, also frequently referred to as the **Performance Index (PI)**, is a composite statistical metric used to evaluate the overall performance of estimation and forecasting models.

It is calculated as the product of the Pearson's Correlation Coefficient (:math:`\text{R}`) and Willmott's Index of Agreement (:math:`\text{WI}`).

.. math::

    \text{CI}(y, \hat{y}) = \text{R}(y, \hat{y}) \times \text{WI}(y, \hat{y})

Note: :math:`\text{R}` measures the linear correlation/phase relationship, while :math:`\text{WI}` measures the degree of error in magnitude and variance. By multiplying them, CI captures both correlation and absolute agreement in a single score.

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Holistic Evaluation:** Models can sometimes cheat individual metrics (e.g., having a high correlation :math:`R` but terrible absolute magnitude, or vice versa). By multiplying :math:`R` and :math:`WI`, CI heavily penalizes models unless they perform well in both trend prediction and magnitude accuracy.
	* **Standardized Benchmarking:** Unlike raw error metrics (MAE, RMSE), CI provides a standardized classification scale, making it extremely easy to categorize model performance for non-technical stakeholders.

**Performance Classification:**
Based on standard hydrological and forecasting literature, CI values are categorized as follows:
	* **> 0.85**: Excellent
	* **0.76 - 0.85**: Very Good
	* **0.66 - 0.75**: Good
	* **0.61 - 0.65**: Satisfactory
	* **0.51 - 0.60**: Poor
	* **0.41 - 0.50**: Bad
	* **< 0.40**: Very Bad

**Disadvantages:**
	* **Negative Value Ambiguity:** Because :math:`\text{WI}` is strictly positive :math:`[0, 1]`, a negative CI score is driven entirely by a negative Pearson :math:`\text{R}`. A negative score simply means the model is inversely correlated with the ground truth, which generally indicates a complete structural failure of the predictive model.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates perfect correlation and perfect agreement).
* **Range:** ``[-1.0, 1.0]`` (Since :math:`\text{R} \in [-1, 1]` and :math:`\text{WI} \in [0, 1]`, their product is strictly bounded between -1 and 1. It does not extend to negative infinity).

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
    # Calculate Confidence Index
    print("CI: ", evaluator.CI())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("CI (Multi-output): ", evaluator.CI(multi_output="raw_values"))
