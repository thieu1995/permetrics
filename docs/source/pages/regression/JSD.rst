JSD - Jensen-Shannon Divergence
===============================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Jensen-Shannon Divergence (JSD)** :cite:`fuglede2004jensen` is a statistical measure used to quantify the similarity between two probability distributions. It is named after mathematicians Johan Jensen and Claude Shannon.

JSD is a symmetric and smoothed version of the Kullback-Leibler Divergence (KLD). Unlike KLD, which is asymmetrical and can approach infinity, JSD is always finite, symmetric (meaning :math:`\text{JSD}(P||Q) = \text{JSD}(Q||P)`), and its square root satisfies the triangle inequality, making it a true mathematical metric.

**Calculation Steps:**

1. Compute the average probability distribution, :math:`M`:

.. math::

    M_i = \frac{1}{2} (y_i + \hat{y}_i)

2. Calculate the Kullback-Leibler Divergence (KLD) between each distribution and :math:`M`:

.. math::

    D_{KL}(y || M) = \sum_{i=1}^{N} y_i \ln\left(\frac{y_i}{M_i}\right)

    D_{KL}(\hat{y} || M) = \sum_{i=1}^{N} \hat{y}_i \ln\left(\frac{\hat{y}_i}{M_i}\right)

3. Compute the JSD as the arithmetic mean of the two KLD values:

.. math::

    \text{JSD}(y || \hat{y}) = \frac{1}{2} D_{KL}(y || M) + \frac{1}{2} D_{KL}(\hat{y} || M)

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Symmetry & Stability:** Because :math:`\text{JSD}(A, B)` equals :math:`\text{JSD}(B, A)`, it acts as a much more reliable distance metric for comparing distributions than KLD.
	* **Bounded Output:** By interpolating the KLD with the mean distribution, JSD successfully avoids the infinite penalty problem when one distribution has a probability of zero where the other does not.

**Disadvantages:**
	* **Strict Domain Constraint:** The input arrays must represent probability distributions or, at the very least, strictly non-negative values. Passing negative numbers will immediately crash the logarithmic calculations.
	* **Not a pure Regression Metric:** JSD is inherently designed for classification probabilities, clustering, and information retrieval (e.g., comparing word frequencies in NLP). Applying it to standard continuous regression targets without prior normalization can yield meaningless results.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Indicates the two distributions are perfectly identical).
* **Range:** ``[0.0, ln(2)]`` (if natural logarithm is used) or ``[0.0, 1.0]`` (if base-2 logarithm is used). It never reaches infinity.
* **Mathematical Reference:** `Machine Learning Mastery <https://machinelearningmastery.com/divergence-between-probability-distributions/>`_

-------------------------------------------------------------------------------

Example Usage
-------------

*Note: Ensure inputs are non-negative, ideally structured as valid probability distributions.*

.. code-block:: python
    :emphasize-lines: 10, 18

    from numpy import array
    from permetrics.regression import RegressionMetric

    ## 1. For 1-D array (Single-output)
    y_true = array([0.1, 0.4, 0.2, 0.3])
    y_pred = array([0.15, 0.35, 0.25, 0.25])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate Jensen-Shannon Divergence
    print("JSD: ", evaluator.JSD())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 0.5], [0.8, 0.2], [0.1, 0.9]])
    y_pred = array([[0.4, 0.6], [0.7, 0.3], [0.2, 0.8]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("JSD (Multi-output): ", evaluator.JSD(multi_output="raw_values"))
