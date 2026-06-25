R2S - Squared Pearson's Correlation Coefficient
===============================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

**Squared Pearson's Correlation Coefficient (R2S / RSQ)** is exactly what the name implies: the mathematical square of Pearson's Correlation Coefficient (R).

In ``permetrics``, this metric is explicitly implemented to combat the widespread nomenclature confusion across online tutorials and major frameworks. Many libraries incorrectly denote the Coefficient of Determination (COD) as :math:`R^2` (R-squared), misleading users into thinking COD is merely the square of the linear correlation. **R2S** serves to physically separate the true squared correlation from the explained variance (COD/R2).

.. math::

    \text{R2S}(y, \hat{y}) = \left[ \frac{\sum_{i=1}^{N} (y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum_{i=1}^{N} (y_i - \bar{y})^2} \sqrt{\sum_{i=1}^{N} (\hat{y}_i - \bar{\hat{y}})^2}} \right]^2

Note: :math:`\bar{y}` and :math:`\bar{\hat{y}}` represent the mean of the actual and predicted values, respectively.

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Educational clarity:** Serves as a definitive benchmark to prove whether a given mathematical framework is calculating the true Coefficient of Determination (explained variance) or merely returning the squared correlation.

**Disadvantages:**
	* **Loss of directional context:** Because the value is squared, it strips away the negative sign of inversely correlated data. A perfect negative correlation (:math:`R = -1`) and a perfect positive correlation (:math:`R = 1`) will both yield an ``R2S`` of ``1.0``.
	* **Redundancy:** In practical modeling, calculating ``R2S`` rarely provides actionable value beyond what the raw Pearson's R already tells you.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Bigger value is better).
* **Range:** ``[0.0, 1.0]``.

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
    # Calculate Squared Pearson's Correlation Coefficient
    print("R2S: ", evaluator.pearson_correlation_coefficient_square())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("R2S (Multi-output): ", evaluator.R2S(multi_output="raw_values"))
