MedAE - Median Absolute Error
=============================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Median Absolute Error (MedAE)** :cite:`nguyen2022improved` is a statistical measure used to evaluate the accuracy of a regression model. Instead of taking the average of the absolute errors (like MAE), it calculates the absolute differences between the predicted and actual values, sorts them, and selects the exact middle value (the median).

.. math::

    \text{MedAE}(y, \hat{y}) = \text{median}(| y_1 - \hat{y}_1 |, | y_2 - \hat{y}_2 |, \ldots, | y_N - \hat{y}_N |)

-------------------------------------------------------------------------------

Description
-----------

**Key Insight: The Ultimate Outlier Resistance**
The core mathematical property of the median is its breakdown point of 50%. This means that up to 50% of your predictions could be infinitely wrong (massive outliers, corrupted data, sensor failures), and the MedAE would remain completely unaffected, steadily reporting the typical error of the "good" half of your data.

**Advantages:**
	* **Outlier Immunity:** While MAE is *robust* to outliers (it treats them linearly), MedAE is entirely *immune* to them (it ignores them completely as long as they represent less than half the dataset). If your dataset is highly prone to extreme anomalies that you explicitly want the evaluation metric to ignore, MedAE is the best choice.
	* **Typical Performance Representation:** It accurately answers the question: *"What is the typical error I can expect on a normal prediction?"* without being skewed by a few catastrophic failures.

**Disadvantages:**
	* **Ignores Tail Risks:** Because it completely ignores the magnitude of the worst errors, a model evaluated solely on MedAE might perform disastrously on edge cases without you ever knowing. It should generally be paired with a metric that penalizes outliers (like RMSE or Max Error) to get a full picture.
	* **Non-Differentiable:** The median operation is not smooth or differentiable, making it highly impractical to use directly as a loss function for training models via gradient descent.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better).
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
    # Calculate Median Absolute Error
    print("MedAE: ", evaluator.MedAE())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("MedAE (Multi-output): ", evaluator.MedAE(multi_output="raw_values"))
