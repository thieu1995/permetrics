CE - Cross Entropy
==================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

**Cross Entropy (CE)**, specifically Binary Cross-Entropy or Log Loss, measures the performance of a model where the prediction input is a probability value between 0 and 1.

It quantifies the difference between two probability distributions: the true distribution (actual labels) and the predicted distribution. As the predicted probability diverges from the actual label, the cross-entropy loss increases exponentially.

.. math::

    \text{CE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Heavy penalization for confident errors:** CE heavily penalizes predictions that are both confident and completely wrong (e.g., predicting 0.99 probability for a class that is actually 0). This forces the model to calibrate its probabilities carefully.
	* **Optimization standard:** Because of its logarithmic nature, it pairs perfectly with sigmoid/softmax activation functions in neural networks, preventing the vanishing gradient problem during backpropagation.

**Disadvantages:**
	* **Strict domain constraint:** The actual values (:math:`y_i`) **must** be strictly binary (0 or 1), and the predicted values (:math:`\hat{y}_i`) **must** be probabilities bounded within (0, 1). Passing raw arbitrary numbers (like temperatures or sales figures) will crash the calculation due to invalid logarithmic domains.
	* **Interpretation difficulty:** Unlike Accuracy or MAE, the absolute value of Cross Entropy is not intuitive to interpret on its own; it is primarily used for comparing models (lower is better).

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Indicates perfect probabilities matching the exact labels).
* **Range:** ``[0, +inf)``
* **Mathematical Reference:** `DataScience StackExchange <https://datascience.stackexchange.com/questions/20296/cross-entropy-loss-explanation>`_

-------------------------------------------------------------------------------

Example Usage
-------------

*Note: Ensure target values are binary (0, 1) and predicted values are probabilities (0.0 to 1.0).*

.. code-block:: python
    :emphasize-lines: 10, 18

    from numpy import array
    from permetrics.regression import RegressionMetric

    ## 1. For 1-D array (Single-output)
    y_true = array([1, 0, 1, 1])
    y_pred = array([0.9, 0.1, 0.8, 0.6])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate Cross Entropy Loss
    print("CE: ", evaluator.CE())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[1, 0], [0, 1], [1, 1]])
    y_pred = array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.6]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("CE (Multi-output): ", evaluator.CE(multi_output="raw_values"))
