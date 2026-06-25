EVS - Explained Variance Score
==============================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Explained Variance Score (EVS)** :cite:`nguyen2020eo` is a regression metric that measures the proportion to which a mathematical model accounts for the variation (dispersion) of a given dataset.

It computes the ratio of the variance of the residual errors (the difference between the true and predicted values) to the variance of the actual true values.

.. math::

    \text{EVS}(y, \hat{y}) = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}

Note: :math:`\text{Var}` denotes the statistical variance, calculated as the average of the squared deviations from the mean.

-------------------------------------------------------------------------------

Description
-----------

**Key Insight**:
	* **EVS vs. R2 (Coefficient of Determination):** While EVS and R2 appear mathematically similar and often yield identical results, they evaluate slightly different concepts. EVS uses the *variance* of the residuals (which removes the mean error), whereas R2 uses the raw sum of squared residuals. If a model is perfectly unbiased (the mean of the errors is exactly 0), EVS will equal R2. However, if a model has a systemic bias, R2 will heavily penalize this bias, whereas EVS will ignore it.

**Advantages:**
	* **Bias-independent variance tracking:** Extremely useful if you want to evaluate how well your model captures the *fluctuations* and trends of the data, even if the absolute calibration (systemic bias) is currently off.

**Disadvantages:**
	* **Ignores systemic bias (Crucial flaw):** A model that consistently predicts values 1000 units higher than the ground truth might still get a perfect EVS of 1.0 if the variance matches perfectly. Therefore, EVS must ALWAYS be paired with a bias metric like Mean Error or CRM to get a complete picture of model accuracy.
	* **Dataset dependency:** Like R2, variance is specific to the scale of the dataset, making cross-dataset EVS comparisons practically meaningless.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates the model perfectly explains the variance of the target variable).
* **Range:** ``(-inf, 1.0]``
* **Mathematical Reference:** `O'Reilly Mastering Python <https://www.oreilly.com/library/view/mastering-python-for/9781789346466/d1ac368a-6890-45eb-b39c-2fa97d23d640.xhtml>`_

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
    # Calculate Explained Variance Score
    print("EVS: ", evaluator.EVS())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("EVS (Multi-output): ", evaluator.EVS(multi_output="raw_values"))
