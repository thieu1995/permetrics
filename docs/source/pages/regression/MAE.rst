MAE - Mean Absolute Error
=========================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Mean Absolute Error (MAE)** :cite:`nguyen2018resource` is a fundamental statistical measure used to evaluate the accuracy of a regression or forecasting model.

It measures the average magnitude of the absolute errors between the predicted values and the actual values, regardless of their direction. Because it uses absolute values, errors do not cancel each other out.

.. math::

    \text{MAE}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|

-------------------------------------------------------------------------------

Description
-----------

**Key Insight: MAE vs. RMSE:**
	The choice between MAE and Root Mean Square Error (RMSE) is one of the most critical decisions in model evaluation. Because MAE uses the L1 Norm (absolute value), it treats all errors linearly. An error of 10 is penalized exactly twice as much as an error of 5. In contrast, RMSE (L2 Norm) squares the errors, meaning an error of 10 is penalized *four times* as much as an error of 5.

**Advantages:**
	* **Outlier Robustness:** Because it does not square the errors, MAE is highly robust to massive outliers. If your dataset contains anomalous extremes that you do not want your model to over-index on, MAE is the superior evaluation metric.
	* **Highly Intuitive:** The output is expressed in the exact same units as the response variable. An MAE of ``5.0`` directly translates to "the model's predictions are off by 5 units on average," which is exceptionally easy to explain to business stakeholders.

**Disadvantages:**
	* **Scale-dependency:** Like RMSE, MAE is completely dependent on the scale of the target variable. You cannot meaningfully compare an MAE of ``100`` (predicting housing prices) with an MAE of ``0.5`` (predicting interest rates).
	* **Optimization difficulty:** The absolute value function is not differentiable at zero, which historically made MAE slightly more complex to use as a direct loss function in gradient descent algorithms compared to MSE (though modern autograd engines handle this seamlessly).

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better, indicating zero error).
* **Range:** ``[0.0, +inf)``

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
    # Calculate Mean Absolute Error
    print("MAE: ", evaluator.MAE())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("MAE (Multi-output): ", evaluator.MAE(multi_output="raw_values"))
