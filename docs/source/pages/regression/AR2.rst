AR2 - Adjusted Coefficient of Determination
===========================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Adjusted Coefficient of Determination** (denoted as **AR2** or **ACOD**) is a modified version of the standard Coefficient of Determination (R2) that adjusts for the number of independent variables (predictors) in a model.

While the standard R2 always increases (or stays the same) when you add new variables—even if they are completely useless—AR2 penalizes you for adding variables that do not genuinely improve the model's predictive power.

.. math::

    \text{AR2}(y, \hat{y}) = 1 - \left[ \frac{(1 - \text{R2})(N - 1)}{N - k - 1} \right]

Where:
	* :math:`\text{R2}` is the standard Coefficient of Determination.
	* :math:`N` is the total number of samples (data points).
	* :math:`k` is the number of independent features/predictors used in the model.

-------------------------------------------------------------------------------

.. note:: Nomenclature Consistency
   As with R2, many external sources and libraries (like Scikit-Learn) refer to this metric as "Adjusted R-squared". Because ``permetrics`` strictly defines R-squared as the literal square of Pearson's correlation, we explicitly denote this metric as **Adjusted R2 (AR2)** or **Adjusted COD** to prevent mathematical misunderstandings.

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Overfitting Prevention:** AR2 is the ultimate safeguard against the "illusion of improvement." It provides a much more accurate and honest measure of goodness-of-fit for multiple regression models by decreasing if a newly added predictor improves the model less than would be expected by chance.
	* **Model Selection:** Highly effective for comparing the explanatory power of regression models that contain different numbers of predictors.

**Disadvantages:**
	* **Sample Size Constraint (Fatal Flaw):** Because the denominator is :math:`N - k - 1`, the metric will mathematically crash (division by zero) if your number of samples equals your number of predictors plus one (:math:`N = k + 1`). Furthermore, if :math:`N < k + 1`, the metric breaks completely. You must always have significantly more data points than features.
	* **Negative Values:** Like standard R2, AR2 can yield negative values if the model is exceptionally poor, which can sometimes confuse non-technical stakeholders who expect a percentage-like score.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Bigger value is better).
* **Range:** ``(-inf, 1.0]``.
* **Mathematical Reference:** `Linear Regression Metrics <https://dziganto.github.io/data%20science/linear%20regression/machine%20learning/python/Linear-Regression-101-Metrics/>`_

-------------------------------------------------------------------------------

Example Usage
-------------

*Note: Depending on your training dataset, ensure that the parameter* ``X_shape`` *is correctly passed in your function called.*

.. code-block:: python
    :emphasize-lines: 10, 18

    from numpy import array
    from permetrics.regression import RegressionMetric

    ## 1. For 1-D array (Single-output)
    y_true = array([3, -0.5, 2, 7])
    y_pred = array([2.5, 0.0, 2, 8])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate Adjusted Coefficient of Determination
    print("AR2: ", evaluator.adjusted_coefficient_of_determination(X_shape=(100, 5))) # 100 samples, 5 features in training dataset

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("AR2 (Multi-output): ", evaluator.AR2(X_shape=(100, 5), multi_output="raw_values"))
