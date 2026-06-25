RAE - Relative Absolute Error
=============================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Relative Absolute Error (RAE)** :cite:`frank2016weka` (WEKA Standard) is a regression metric that compares the absolute error of your model to the absolute error of a simple "no-knowledge" baseline model (which simply predicts the mean of the actual values).

It is calculated as the ratio of the total absolute error of the model to the total absolute deviation of the actual values from their mean.

.. math::

    \text{RAE}(y, \hat{y}) = \frac{\sum_{i=1}^{N} |y_i - \hat{y}_i|}{\sum_{i=1}^{N} |y_i - \bar{y}|}

Note: :math:`\bar{y}` represents the mean of the actual observed values.

-------------------------------------------------------------------------------

.. warning:: The Misunderstanding of "Absolute" vs. "Squared" Formulas
   Many online tutorials and blogs incorrectly present the formula for RAE using squared differences and square roots (e.g., :math:`\frac{\sqrt{\sum(\hat{y}_i - y_i)^2}}{\sqrt{\sum y_i^2}}`). This is a severe mathematical nomenclature error. That formula actually describes a variation of the **Root Relative Squared Error (RRSE)** or numerical linear algebra error norms.

   In data science, the term "Absolute" strictly mandates the use of the L1 Norm (absolute values :math:`|x|`), ensuring linear penalization of errors and outlier resilience. Using squared terms (L2 Norm) amplifies outliers and destroys the core properties of an absolute metric. ``permetrics`` strictly implements the mathematically pure L1 formula, aligning with academic gold standards like the WEKA Machine Learning toolkit.

-------------------------------------------------------------------------------

Description
-----------

**Advantages:** RAE provides a very clear benchmark for model performance:
    * ``RAE < 1.0``: Your model performs *better* than the simple baseline (predicting the mean).
    * ``RAE = 1.0``: Your model is exactly as accurate as just predicting the mean.
    * ``RAE > 1.0``: Your model is *worse* than the simple baseline.
* **Scale-independent:** Because the error is divided by the data's inherent deviation, the metric is dimensionless. This allows you to compare the predictive accuracy of models across entirely different datasets with different scales.
* **Outlier resilience:** Because it uses absolute differences rather than squared differences, RAE does not heavily exaggerate the impact of single extreme outliers.

**Disadvantages:**
	* **Zero-Deviation Vulnerability:** If all actual values in the ground truth dataset are identical (meaning there is zero variance and :math:`y_i = \bar{y}` for all :math:`i`), the denominator becomes exactly zero. This will cause the formula to crash (division by zero) or return an undefined/NaN value.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better, meaning zero absolute error).
* **Range:** ``[0.0, +inf)``
* **Mathematical Reference:** `WEKA Machine Learning Evaluation <https://waikato.github.io/weka-wiki/documentation/>`_

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
    # Calculate Relative Absolute Error
    print("RAE: ", evaluator.RAE())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("RAE (Multi-output): ", evaluator.RAE(multi_output="raw_values"))
