MBE - Mean Bias Error
=====================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Mean Bias Error (MBE)** :cite:`kato2016prediction` is a fundamental statistical measure used to evaluate the systematic bias of a forecasting model. It calculates the average difference between the predicted values and the actual values while strictly preserving the sign (direction) of the errors.

.. math::

    \text{MBE}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)

Note: In ``permetrics``, the formula is calculated as Predicted minus Actual :math:`(\hat{y}_i - y_i)`. Therefore, a positive MBE strictly indicates that the model tends to overestimate, while a negative MBE indicates underestimation.

-------------------------------------------------------------------------------

Description
-----------

**Key Insight: The Cancellation Effect (Fatal Flaw as an Accuracy Metric)**
Because MBE preserves the direction of errors, positive errors (over-predictions) and negative errors (under-predictions) will cancel each other out. A model that predicts :math:`+100` for the first sample and :math:`-100` for the second sample will have a perfect MBE of ``0.0``, masking the fact that its individual predictions are highly inaccurate. Therefore, MBE is **never** a measure of absolute accuracy. It must always be paired with an absolute metric like MAE or RMSE.

**Advantages:**
	* **Directional Diagnostic:** It is the ultimate diagnostic tool to determine if your model is systematically over-forecasting or under-forecasting.
	* **Conservation of Mass/Energy:** In physical sciences and inventory management, keeping MBE close to zero ensures that the *total volume* predicted over a period matches the *total volume* observed, even if individual days are inaccurate.

**Disadvantages:**
	* **Illusion of Perfection:** An MBE of zero does not mean the model is perfect; it simply means the sum of over-predictions perfectly balances the sum of under-predictions.
	* **Outlier Sensitivity:** Because it uses raw differences, an extreme outlier in one direction can heavily skew the bias. If data is heavily skewed, the **Median Bias Error (MdBE)** is often a more robust alternative.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Indicates zero systematic bias; over-predictions perfectly balance under-predictions).
* **Range:** ``(-inf, +inf)``
    * **MBE > 0:** The model systematically overestimates.
    * **MBE < 0:** The model systematically underestimates.

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

    # Calculate Mean Bias Error
    print("MBE: ", evaluator.MBE())
    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("MBE (Multi-output): ", evaluator.MBE(multi_output="raw_values"))
