MAAPE - Mean Arctangent Absolute Percentage Error
=================================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Mean Arctangent Absolute Percentage Error (MAAPE)** is a robust forecasting metric introduced by :cite:`kim2016new` specifically to overcome the fatal flaws of the traditional Mean Absolute Percentage Error (MAPE).

It applies the inverse tangent (arctangent) function to the absolute percentage error. Because the arctangent function mathematically compresses all input values into a finite range, it inherently protects the metric from exploding to infinity.

.. math::

    \text{MAAPE}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} \arctan\left( \left| \frac{y_i - \hat{y}_i}{y_i} \right| \right)

Note: The result is measured in radians. Some implementations multiply this by 100 to pseudo-scale it, but the mathematically pure form remains unscaled.

-------------------------------------------------------------------------------

Description
-----------

**Key Insight: The Infinity Compression (Zero-Value Solution)**
The greatest weakness of MAPE is that if the actual value :math:`y_i = 0`, the division yields infinity (:math:`\infty`), crashing the system. MAAPE brilliantly solves this using trigonometry. In programming (like NumPy), dividing by zero yields ``Inf``. The arctangent of infinity is exactly :math:`\pi/2` (approx. 1.5708). Thus, if the ground truth is zero, the error smoothly caps out at 1.5708 instead of crashing, making MAAPE the ultimate metric for intermittent demand forecasting (where zero-sales days are common).

**Advantages:**
	* **Zero-Value Immunity:** Perfectly handles datasets with zero values without needing artificial epsilon additions or data masking.
	* **Symmetric Bounding:** By applying the arctangent function, the extreme asymmetry of MAPE (where over-predictions can cause unbounded >1000% errors) is flattened out. A massive over-prediction simply pushes the error term asymptotically closer to :math:`\pi/2`.

**Disadvantages:**
	* **Loss of Intuitive Interpretation:** Unlike MAPE, where ``0.20`` means "off by 20%", a MAAPE score of ``0.85`` cannot be directly translated into a clear business percentage. It exists in a non-linear radian space.
	* **Under-penalization of extremes:** Because of the asymptotic nature of the arctangent curve, the penalty difference between an error of 500% and an error of 10000% is mathematically minuscule.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better, indicating zero error).
* **Worst possible score:** Approaches ``1.5708`` (:math:`\pi/2`). This occurs when predictions are infinitely wrong or when evaluating against a true value of zero.
* **Range:** ``[0.0, \pi/2)`` (Strictly bounded, it NEVER reaches infinity).
* **Mathematical Reference:** `NumXL MAAPE Documentation <https://support.numxl.com/hc/en-us/articles/115001223463-MAAPE-Mean-Arctangent-Absolute-Percentage-Error>`_

-------------------------------------------------------------------------------

Example Usage
-------------

Note: If your dataset contains zeros, ensure your matrix operations safely yield `np.inf` rather than throwing fatal exceptions, allowing `np.arctan` to handle the bounding.

.. code-block:: python
    :emphasize-lines: 10, 18

    from numpy import array
    from permetrics.regression import RegressionMetric

    ## 1. For 1-D array (Single-output)
    y_true = array([3, 0.0, 2, 7])
    y_pred = array([2.5, 0.5, 2, 8])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate Mean Arctangent Absolute Percentage Error
    print("MAAPE: ", evaluator.MAAPE())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [0.0, 1], [7, 6]])
    y_pred = array([[0.6, 2], [0.1, 2], [8, 5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("MAAPE (Multi-output): ", evaluator.MAAPE(multi_output="raw_values"))
