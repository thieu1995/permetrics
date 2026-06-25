SLE - Squared Log Error
=======================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

**Squared Log Error (SLE)** is an element-wise metric that measures the squared difference between the natural logarithm of the predicted values and the natural logarithm of the actual (ground truth) values.

By applying a logarithmic transformation before calculating the squared error, SLE inherently penalizes underestimates more heavily than overestimates and focuses on the relative (percentage) difference rather than the absolute scale.

.. math::

    \text{SLE}_i = (\log(y_i + 1) - \log(\hat{y}_i + 1))^2

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Outlier dampening:** The logarithmic scale compresses massive values, making SLE highly resilient to extreme outliers. It is the ideal metric when targets have an exponential growth trend or right-skewed distribution (e.g., population growth, housing prices).
	* **Relative deviation:** It measures the ratio of the true to the predicted value. An error between predicting 10 for a true value of 20 is heavily penalized, while an error between 1010 and 1020 is treated as practically negligible.

**Disadvantages:**
	* **Domain restriction (Crucial):** Because the mathematical formulation uses :math:`\log(x + 1)`, the input values for both actual and predicted data **must be strictly greater than -1**. If your dataset contains negative values (e.g., temperatures below zero), SLE will return undefined bounds (NaN/Inf) and crash your evaluation pipeline.
	* **Non-aggregated:** SLE computes an array of element-wise errors. To evaluate the entire model, it must be aggregated into Mean Squared Logarithmic Error (MSLE).

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better).
* **Range:** ``[0, +inf)``

-------------------------------------------------------------------------------

Example Usage
-------------

*Note: Ensure all target and predicted values are > -1 to avoid math domain errors.*

.. code-block:: python
    :emphasize-lines: 10, 18

    from numpy import array
    from permetrics.regression import RegressionMetric

    ## 1. For 1-D array (Single-output)
    y_true = array([3, 0.5, 2, 7])
    y_pred = array([2.5, 0.0, 2, 8])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate Squared Log Error for each element
    print("SLE: ", evaluator.SLE())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [1.5, 1], [7, 6]])
    y_pred = array([[0, 2], [1.0, 2], [8, 5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate Squared Log Error for multi-dimensional array
    print("SLE (Multi-output): ", evaluator.SLE())
