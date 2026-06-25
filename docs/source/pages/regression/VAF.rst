VAF - Variance Accounted For
============================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Variance Accounted For (VAF)** is a metric heavily utilized in signal processing and control systems (e.g., system identification) to evaluate how well a model captures the dynamic behavior of an actual signal.

Mathematically, it represents the percentage of the true signal's variance that is successfully predicted by the model. It is effectively the Explained Variance Score (EVS) expressed as a percentage.

.. math::

    \text{VAF}(y, \hat{y}) = \left( 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)} \right) \times 100\%

Note: :math:`\text{Var}` denotes the statistical variance. If the mean of the actual data is zero, VAF is mathematically equivalent to :math:`R^2 \times 100\%`.*

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Signal Processing Standard:** It is the industry-standard benchmark for evaluating time-series tracking and system identification models (e.g., widely used in MATLAB's System Identification Toolbox).
	* **Intuitive Percentage Scale:** Expressing accuracy as a percentage (up to 100%) is highly interpretable for clients, engineers, and non-technical stakeholders compared to raw decimal scores like R2 or EVS.

**Disadvantages:**
	* **Redundancy in standard ML:** In general Machine Learning applications, VAF offers no new mathematical information beyond what EVS or R2 already provides.
	* **Negative percentages:** Just like R2 or EVS, if a model performs catastrophically worse than simply predicting the mean, the VAF can drop into negative percentages (e.g., -250%), which can sometimes confuse users who expect a strict 0-100% scale.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``100.0`` (Indicates the two signals are perfectly identical).
* **Range:** ``(-inf, 100.0]``
* **Mathematical Reference:** `TU Delft LTI Toolbox <https://www.dcsc.tudelft.nl/~jwvanwingerden/lti/doc/html/vaf.html>`_

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
    # Calculate Variance Accounted For
    print("VAF: ", evaluator.VAF())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("VAF (Multi-output): ", evaluator.VAF(multi_output="raw_values"))
