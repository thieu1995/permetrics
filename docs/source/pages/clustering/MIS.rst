MIS - Mutual Information Score
==============================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Mutual Information Score (MIS)** :cite:`cover2006elements` is an external clustering evaluation metric that quantifies the "amount of information" (in nats) shared between the ground truth labels (:math:`y_{true}`) and the predicted labels (:math:`y_{pred}`).

Intuitively, MIS measures the reduction in uncertainty about one clustering partition given knowledge of the other. If the two partitions are identical, MIS reaches its maximum; if they are independent, MIS is zero. Unlike metrics based on pair counting (like Rand Score), MIS is based on information theory and is invariant to permutations of cluster labels.

.. math::

    \text{MIS}(Y, P) = \sum_{y \in Y} \sum_{p \in P} P(y, p) \log \left( \frac{P(y, p)}{P(y)P(p)} \right)

Where:

* :math:`Y` is the set of ground truth classes.
* :math:`P` is the set of predicted clusters.
* :math:`P(y, p)` is the joint probability of a sample belonging to class :math:`y` and cluster :math:`p`.
* :math:`P(y)` and :math:`P(p)` are the marginal probabilities.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** No strict upper bound (the score is upper-bounded by the entropy of the partition).
* **Worst possible score:** ``0.0`` (Indicates the two partitions are completely independent).
* **Range:** ``[0.0, +inf)``
* **References:** `Scikit-Learn Mutual Information <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 13,15,24,26

    from permetrics.clustering import ClusteringMetric
    import numpy as np

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC MUTUAL INFORMATION SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    
    # Initialize the metric object
    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    # Calculate the Mutual Information Score
    mis_score = cm.MIS()
    print(f"Mutual Information Score: {mis_score}")

    # ==============================================================================
    # SCENARIO 2: Perfect Agreement vs. Random Partition
    # ==============================================================================
    print("\n--- 2. AGREEMENT ANALYSIS ---")

    # Perfect match
    print(f"Perfect Agreement: {cm.MIS(y_true=[0, 0, 1], y_pred=[0, 0, 1])}")
    # Random partition
    print(f"Random Partition:  {cm.MIS(y_true=[0, 0, 1], y_pred=[1, 0, 0])}")
