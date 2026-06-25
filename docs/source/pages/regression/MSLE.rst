MSLE - Mean Squared Logarithmic Error
=====================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Mean Squared Logarithmic Error (MSLE)** :cite:`hodson2021mean` is a variation of the Mean Squared Error (MSE) that incorporates a logarithmic transformation.

It measures the expected value of the squared logarithmic error, making it highly suitable for targets with exponential growth patterns, such as population counts, total sales, or web traffic.

.. math::

    \text{MSLE}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} \left( \ln(1 + y_i) - \ln(1 + \hat{y}_i) \right)^2

Note: :math:`\ln` denotes the natural logarithm. The addition of :math:`1` to both :math:`y` and :math:`\hat{y}` ensures the mathematical safety of the logarithm when dealing with zero values.

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Relative Error Focus:** By applying the logarithm, MSLE treats small differences between small true and predicted values approximately the same as big differences between large true and predicted values. It focuses on the *relative ratio* rather than the absolute difference.
	* **Asymmetric Penalization (Crucial Feature):** Unlike standard MSE, MSLE strongly penalizes *under-predictions* more than *over-predictions*. If actual sales are 1000, predicting 600 (under) will yield a significantly higher MSLE penalty than predicting 1400 (over). This is ideal for inventory domains where under-stocking is costlier than over-stocking.
	* **Zero-Value Safe:** The :math:`\ln(1+x)` transformation perfectly handles cases where the actual or predicted value is exactly ``0.0``.

**Disadvantages:**
	* **Strict Domain Constraint:** Actual and predicted values **must** be strictly non-negative (:math:`\ge 0`). If the model outputs a negative prediction (or if the dataset contains negative values), the calculation will crash or return NaN due to an invalid logarithmic domain.
	* **Interpretation:** The absolute value of MSLE is not intuitive on its own; it is primarily used for comparing the relative performance of different models.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better).
* **Range:** ``[0.0, +inf)``

-------------------------------------------------------------------------------

Example Usage
-------------

*Note: Ensure that all target and predicted values are non-negative.*

.. code-block:: python
    :emphasize-lines: 10, 18

    from numpy import array
    from permetrics.regression import RegressionMetric

    ## 1. For 1-D array (Single-output)
    y_true = array([3, 0.5, 2, 7])
    y_pred = array([2.5, 0.0, 2, 8])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate Mean Squared Logarithmic Error
    print("MSLE: ", evaluator.MSLE())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [1, 1], [7, 6], [1, 2]])
    y_pred = array([[0, 2], [1, 2], [8, 5], [1.1, 1.9]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("MSLE (Multi-output): ", evaluator.MSLE(multi_output="raw_values"))
