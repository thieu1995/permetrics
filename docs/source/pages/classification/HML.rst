HS - Hamming Score
==================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Hamming Score** (also widely known as **Multi-Label Accuracy**) is a dedicated evaluation metric for multi-label classification tasks. Unlike standard single-label accuracy which strictly demands a perfect match across all target labels, the Hamming Score computes the fractional correctness by taking the ratio of the intersection to the union of the predicted and true label sets for each sample, then averaging across the entire dataset.

.. image:: /_static/images/class_score_1.png
   :align: center
   :alt: Hamming Score Multi-label Illustration

.. math::

    \text{Hamming Score} = \frac{1}{N} \sum_{i=1}^{N} \frac{|y_i \cap \hat{y}_i|}{|y_i \cup \hat{y}_i|}

Where:

* :math:`y_i` is the set of true labels for the :math:`i`-th sample.
* :math:`\hat{y}_i` is the set of predicted labels for the :math:`i`-th sample.
* :math:`N` is the total number of samples.

-------------------------------------------------------------------------------

Description
-----------

**Key Insight: Partial Credit in Multi-Label Learning**
In multi-label classification, a single instance can belong to multiple classes simultaneously (e.g., an article tagged as "Politics", "Economics", and "Europe"). If a model predicts "Politics" and "Economics" but misses "Europe", standard subset accuracy scores this sample as a complete failure (``0.0``). Hamming Score solves this by awarding a partial credit of ``2/3 = 0.67`` because it directly evaluates the semantic overlap between the prediction and the ground truth.

**Advantages:**
* **Soft Evaluation Strategy:** Highly fair and granular for multi-label environments, as it rewards models that get the majority of labels right instead of penalizing near-misses in a strict binary fashion.
* **Intuitive Range:** Bounded perfectly between ``0.0`` and ``1.0``, making it directly comparable to traditional accuracy metrics.

**Disadvantages:**
* **Label Sparsity Sensitivity:** On datasets where active labels are highly sparse (mostly zeros), the union (denominator) can become very small, causing the score to be highly volatile if the model over-predicts false positives.
* **Degradation to Accuracy:** In standard single-label multiclass problems, the intersection over union simplifies completely, causing the Hamming Score to degrade into standard Accuracy, offering no unique analytical advantage in that domain.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates a perfect prediction where the predicted label set matches the true label set exactly for every single sample).
* **Worst possible score:** ``0.0`` (Occurs when there is absolutely zero overlap between the predicted and true labels across all samples).
* **Range:** ``[0.0, 1.0]``
* **References:** * `Tsoumakas, G., & Katakis, I. (2007). Multi-label classification: An overview. <https://www.researchgate.net/publication/220519965_Multi-Label_Classification_An_Overview>`_
    * `Sorower, M. S. (2010). A literature survey on algorithms for multi-label learning. <https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.364.7176>`_

-------------------------------------------------------------------------------

Example Usage
-------------

*Note: In ``permetrics``, the input for multi-label metrics expects binary indicator matrices (0 or 1) representing label presence across samples.*

.. code-block:: python
    :emphasize-lines: 11-13, 18, 28

    from permetrics.classification import ClassificationMetric
    import numpy as np

    # ==============================================================================
    # SCENARIO 1: Multi-Label Classification (Binary Indicator Matrices)
    # Samples can have multiple active tags at the same time
    # ==============================================================================
    print("--- 1. MULTI-LABEL CLASSIFICATION EXAMPLES ---")

    # 4 samples, 3 possible classes/labels
    y_true_ml = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 0, 1]])
    y_pred_ml = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 0, 0]])

    cm_ml = ClassificationMetric(y_true_ml, y_pred_ml)

    # Calculate Hamming Score
    hs_score = cm_ml.HS()
    print(f"Hamming Score (Multi-label Accuracy): {hs_score}")

    # ==============================================================================
    # SCENARIO 2: Single-Label Multiclass Boundary Case
    # In single-label cases, Hamming Score yields the same result as standard Accuracy
    # ==============================================================================
    print("\n--- 2. SINGLE-LABEL BOUNDARY CASE ---")

    y_true_sl = [0, 1, 2, 0]
    y_pred_sl = [0, 2, 2, 0]

    cm_sl = ClassificationMetric(y_true_sl, y_pred_sl)
    print(f"Hamming Score on Single-label data : {cm_sl.HS()}")
