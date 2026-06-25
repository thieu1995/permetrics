COV - Covariance
================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

**Covariance (COV)** is a statistical measure that evaluates the extent to which two variables—the actual values (:math:`y`) and the predicted values (:math:`\hat{y}`)—change together.

It determines the directional relationship between the predictions and the ground truth, revealing whether they tend to increase or decrease in tandem.

**Population Covariance:**

.. math::

    \text{COV}_{\text{pop}}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \bar{y}) (\hat{y}_i - \bar{\hat{y}})

**Sample Covariance (Bessel's correction):**

.. math::

    \text{COV}_{\text{samp}}(y, \hat{y}) = \frac{1}{N-1} \sum_{i=1}^{N} (y_i - \bar{y}) (\hat{y}_i - \bar{\hat{y}})

Note: :math:`\bar{y}` and :math:`\bar{\hat{y}}` represent the mean of the actual and predicted values, respectively.

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Directional insight:** A positive covariance indicates that as actual values increase, predicted values also tend to increase (moving in the same direction). A negative covariance reveals an inverse relationship.
	* **Mathematical foundation:** It serves as the core building block for calculating more advanced and interpretable metrics, such as Pearson's Correlation Coefficient (COR).

**Disadvantages:**
	* **Scale-dependency (Crucial limitation):** Unlike Correlation, Covariance is *not* normalized. Its magnitude depends entirely on the units of the data. You cannot compare the COV of a dataset measured in millimeters with one measured in kilometers.
	* **Ignores magnitude of error:** COV only measures whether variables move together, not how close the predictions actually are to the ground truth. It does not assess the absolute accuracy of a model.
	* **Unbounded:** Because it has no upper or lower limits, a standalone covariance score is nearly impossible to interpret without additional context.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``Undefined`` (There is no absolute "best" value. Larger positive/negative values simply indicate a stronger directional trend relative to the specific dataset's scale).
* **Range:** ``(-inf, +inf)``
* **Mathematical Reference:** `Corporate Finance Institute <https://corporatefinanceinstitute.com/resources/data-science/covariance/>`_

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
    # Calculate Covariance
    print("COV: ", evaluator.COV())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("COV (Multi-output): ", evaluator.COV(multi_output="raw_values"))
