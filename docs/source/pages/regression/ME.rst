ME - Max Error
==============

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Max Error (ME)** (also known as Maximum Residual Error) is a regression metric that captures the absolute worst-case error between the predicted values and the true values.

Unlike average-based metrics (like MAE or RMSE) that aggregate errors across the entire dataset, Max Error strictly isolates and reports the single largest deviation.

.. math::

    \text{ME}(y, \hat{y}) = \max_{i=1}^{N} (| y_i - \hat{y}_i |)

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Safety-Critical Benchmarking:** This is the ultimate evaluation metric for high-stakes environments (e.g., industrial sensor calibration, clinical dosage prediction, autonomous driving) where bounding the absolute maximum failure is far more critical than average performance. If a single severe error can break a machine or harm a patient, ME is the metric you must monitor.
	* **Worst-Case Guarantee:** It establishes a strict upper bound on the model's error. If your ME is ``2.0``, you can mathematically guarantee that no prediction in your test set was off by more than ``2.0``.

**Disadvantages:**
	* **Extreme Outlier Vulnerability:** By definition, ME focuses entirely on a single data point. A single corrupted ground-truth label, a broken sensor reading, or an unpredictable anomaly will cause the ME to explode. This can falsely make a model look terrible, hiding the fact that it might be perfectly fitted to the other 99.9% of the data.
	* **Not Differentiable:** Because it relies on the strict maximum operator, it is generally not smooth and cannot be used directly as a loss function for training neural networks via gradient descent.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Indicates a perfectly fitted model with zero error across all data points).
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
    # Calculate Max Error
    print("ME: ", evaluator.max_error())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("ME (Multi-output): ", evaluator.ME(multi_output="raw_values"))
