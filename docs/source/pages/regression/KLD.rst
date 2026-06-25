KLD - Kullback-Leibler Divergence
=================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Kullback-Leibler Divergence (KLD)** :cite:`hershey2007approximating`, also known as relative entropy, is a foundational statistical measure originating from information theory. It quantifies how much one probability distribution (the predictions, :math:`\hat{y}`) differs from a reference probability distribution (the ground truth, :math:`y`).

.. math::

    D_{KL}(y || \hat{y}) = \sum_{i=1}^{N} y_i \ln\left(\frac{y_i}{\hat{y}_i}\right)

Note: :math:`\ln` denotes the natural logarithm. The formula calculates the expectation of the logarithmic difference between the probabilities.

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Information Loss Measurement:** KLD is exceptional at measuring the exact amount of "information lost" when you use the predicted distribution to approximate the true distribution.
	* **Optimization Standard:** It is the core mathematical engine behind Cross-Entropy Loss (minimizing Cross-Entropy is directly tied to minimizing KLD), making it ubiquitous in machine learning and neural network training.

**Disadvantages:**
	* **Asymmetry (Crucial Limitation):** KLD is **not** a true statistical distance metric because it is inherently asymmetric. :math:`D_{KL}(A || B)` does not equal :math:`D_{KL}(B || A)`. It also does not satisfy the triangle inequality. If you need a symmetric distance, use the Jensen-Shannon Divergence (JSD).
	* **Zero-Probability Crash:** The formula divides by :math:`\hat{y}_i`. If your model predicts exactly ``0.0`` for an event that actually occurs in the ground truth (:math:`y_i > 0`), the formula involves division by zero and will explode to infinity. *(Implementation note: Always add a tiny epsilon to the denominator).*
	* **Strict Domain Constraint:** Both arrays must strictly contain non-negative values (ideally representing valid probability distributions where the sum equals 1).

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Indicates the two distributions are perfectly identical).
* **Range:** ``[0.0, +inf)`` (By Gibbs' inequality, KLD is always non-negative).
* **Mathematical Reference:** `Machine Learning Mastery <https://machinelearningmastery.com/divergence-between-probability-distributions/>`_

-------------------------------------------------------------------------------

Example Usage
-------------

*Note: Ensure inputs are strictly non-negative, ideally structured as valid probability distributions.*

.. code-block:: python
    :emphasize-lines: 10, 18

    from numpy import array
    from permetrics.regression import RegressionMetric

    ## 1. For 1-D array (Single-output)
    y_true = array([0.1, 0.4, 0.2, 0.3])
    y_pred = array([0.15, 0.35, 0.25, 0.25])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate Kullback-Leibler Divergence
    print("KLD: ", evaluator.KLD())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 0.5], [0.8, 0.2], [0.1, 0.9]])
    y_pred = array([[0.4, 0.6], [0.7, 0.3], [0.2, 0.8]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("KLD (Multi-output): ", evaluator.KLD(multi_output="raw_values"))
