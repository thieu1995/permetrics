RaS - Rand Score
================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Rand Score (RaS)** is a fundamental external clustering evaluation metric. It measures the similarity between two clustering partitions by considering all pairs of samples and counting pairs that are assigned in the same or different clusters across both partitions.

Intuitively, RaS operates identically to accuracy in classification, applied to pairs of data points. It answers the question: *"Of all possible pairs of samples, what percentage were categorized agreeably (either grouped together or separated) across both partitions?"*

.. math::

    \text{RaS} = \frac{yy + nn}{yy + yn + ny + nn} = \frac{yy + nn}{\binom{N}{2}}

Where, across all :math:`\binom{N}{2}` possible pairs of distinct data points:

* :math:`yy` (True Positives): Number of pairs placed in the **same** cluster in both :math:`y_{true}` and :math:`y_{pred}`.
* :math:`nn` (True Negatives): Number of pairs placed in **different** clusters in both :math:`y_{true}` and :math:`y_{pred}`.
* :math:`yn` and :math:`ny` are the discordant pairs (False Negatives and False Positives).

-------------------------------------------------------------------------------

Algorithmic Optimizations (Performance Note)
--------------------------------------------

Standard pair counting requires explicitly generating all combinations, resulting in an expensive space and time complexity of :math:`O(N^2)`. 

This implementation computes the exact Rand Score in **:math:`O(N)` time complexity** by deriving the pair distributions directly from the algebraic properties of the **Contingency Matrix** (using dot products of its marginal sums). This allows seamless benchmarking on massive datasets without memory bottlenecks.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates absolute agreement between partitions).
* **Worst possible score:** ``0.0`` (Indicates zero agreement; no pair of points shares a cohesive grouping).
* **Permutation Invariance:** The score is completely invariant to permutations of cluster labels (e.g., partitioning ``[0, 0, 1]`` vs ``[1, 1, 0]`` yields a score of 1.0).
* **Range:** ``[0.0, 1.0]``
* **Known Limitation:** The raw Rand Score approaches 1.0 as the number of clusters increases, even for completely random clusterings. For a chance-corrected variant, see the **Adjusted Rand Score (ARS)**.
* **References:** `Scikit-Learn Rand Score Documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.rand_score.html>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11,12,21,22

    from permetrics.clustering import ClusteringMetric

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC RAND SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 2, 1, 2]
    
    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    ras_score = cm.RaS()
    print(f"Rand Score: {ras_score}")

    # ==============================================================================
    # SCENARIO 2: Demonstrating Permutation Invariance
    # ==============================================================================
    print("\n--- 2. PERMUTATION INVARIANCE EXAMPLE ---")

    # Cluster names are flipped (0->1, 1->0), but the structure is identical
    cm_flipped = ClusteringMetric(y_true=[0, 0, 1, 1], y_pred=[1, 1, 0, 0])
    print(f"Flipped Labels RaS: {cm_flipped.RaS()}")  # Returns 1.0
