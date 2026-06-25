RSE - Residual Standard Error
=============================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Residual Standard Error (RSE)** is a statistical metric used to evaluate the goodness-of-fit of a regression model. It measures the sample standard deviation of the residual errors, representing the average absolute amount that the actual values deviate from the true regression line.

.. math::

    \text{RSE}(y, \hat{y}) = \sqrt{\frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{N - k - 1}}

Where:
	* :math:`N` is the total number of samples (data points).
	* :math:`k` is the number of independent features/predictors used in the model.

-------------------------------------------------------------------------------

Description
-----------

**Key Insight:**
	* **RSE vs. RMSE:** While Root Mean Square Error (RMSE) divides the sum of squared errors by :math:`N`, RSE divides it by the degrees of freedom (:math:`N - k - 1`). This makes RSE an mathematically unbiased estimator of the standard deviation of the error term (:math:`\sigma`). In large datasets, RSE and RMSE will be nearly identical, but in small datasets with many predictors, RSE provides a much more honest penalty for model complexity.

**Advantages:**
	* **Unbiased Estimation:** It is the gold standard for statistical inference (e.g., calculating confidence intervals and p-values for regression coefficients) because it accounts for the degrees of freedom lost by estimating the model's parameters.
	* **Heavy Outlier Penalty:** Like RMSE, it squares the errors, making it highly sensitive to massive deviations.

**Disadvantages:**
	* **Scale-dependency:** The output is expressed in the exact same units as the response variable. While intuitive for a single dataset, it makes comparing RSE scores across datasets with different scales impossible.
	* **Sample Size Constraint (Crucial Flaw):** Just like Adjusted R2, the denominator is :math:`N - k - 1`. If the number of predictors (:math:`k`) is too large relative to your sample size, the metric will crash (division by zero) or become mathematically invalid. You must always have more data points than features.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better, indicating zero residual error).
* **Range:** ``[0.0, +inf)``
* **Mathematical Reference:** `Statology <https://www.statology.org/residual-standard-error-r/>`_ and `Machine Learning Mastery (Degrees of Freedom) <https://machinelearningmastery.com/degrees-of-freedom-in-machine-learning/>`_

-------------------------------------------------------------------------------

Example Usage
-------------

*Note: Depending on your training dataset, ensure that the parameter* ``X_shape`` is correctly passed to the function.*

.. code-block:: python
    :emphasize-lines: 10, 18

    from numpy import array
    from permetrics.regression import RegressionMetric

    ## 1. For 1-D array (Single-output)
    y_true = array([3, -0.5, 2, 7])
    y_pred = array([2.5, 0.0, 2, 8])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate Residual Standard Error
    print("RSE: ", evaluator.RSE(X_shape=(200, 5)))	# (Number of samples, number of features)

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("RSE (Multi-output): ", evaluator.RSE(X_shape=(200, 5), multi_output="raw_values"))
