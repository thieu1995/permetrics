SI - Silhouette Index
=====================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Silhouette Index (SI)** :cite:`rousseeuw1987silhouettes` is a highly popular internal clustering evaluation metric. It measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation).

Intuitively, SI answers the question: *"How well does each data point fit into its assigned cluster compared to the next best alternative cluster?"* A high silhouette value indicates that the object is well-matched to its own cluster and poorly matched to neighboring clusters.

.. math::

    s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}

Where:

* :math:`a(i)` is the mean distance between data point :math:`i` and all other points in the same cluster (intra-cluster distance).
* :math:`b(i)` is the smallest mean distance between data point :math:`i` and all points in any other cluster, of which :math:`i` is not a member (nearest-cluster distance).
* The global Silhouette Index is the mean of the silhouette widths :math:`s(i)` for all data points.

-------------------------------------------------------------------------------

Algorithmic Variations (Memory Optimization)
--------------------------------------------

Calculating the exact Silhouette Score normally requires instantiating a full distance matrix, which has a space complexity of :math:`O(N^2)`. To prevent Out-Of-Memory (OOM) errors on large datasets (e.g., :math:`N > 100,000`), this implementation utilizes a highly optimized **chunk-based processing** strategy.

* **chunk_size (int):** Processes the pairwise distances in bounded batches (default: ``5000``). This tightly caps the RAM usage to a safe limit while mathematically guaranteeing the exact same result as the standard approach.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The Silhouette Index is mathematically undefined when there is only one cluster (:math:`K = 1`). In this scenario, neither cohesion nor separation can be fully established.

* **force_finite (bool):** If ``True``, catches the undefined operation and returns a safe fallback. Default is ``True``.
* **finite_value (float):** The specific fallback value returned when ``force_finite=True``. Since the worst possible silhouette score is -1, the default fallback is a penalty value of ``-1.0``.
* **multi_output (bool):** If ``True``, the function returns an array of silhouette scores for each individual data point instead of a single global mean.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Points are perfectly clustered and far away from neighboring clusters).
* **Worst possible score:** ``-1.0`` (Points are consistently assigned to the wrong clusters).
* **Values near 0:** Indicate overlapping clusters where points are situated on the decision boundary between two groups.
* **Range:** ``[-1.0, 1.0]``
* **References:** `Scikit-Learn Silhouette Score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 12,14,23,32,35

    from permetrics.clustering import ClusteringMetric
    import numpy as np

    # ==============================================================================
    # SCENARIO 1: Normal Evaluation (Global Mean Silhouette)
    # ==============================================================================
    print("--- 1. BASIC SILHOUETTE INDEX EXAMPLE ---")

    X_data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    y_pred_labels = np.array([0, 0, 0, 1, 1, 1])
    
    cm = ClusteringMetric(X=X_data, y_pred=y_pred_labels)
    # Calculates the global mean silhouette score
    si_score = cm.SI()
    print(f"Silhouette Index: {si_score}")

    # ==============================================================================
    # SCENARIO 2: Multi-output (Per-sample Silhouette Scores)
    # ==============================================================================
    print("\n--- 2. PER-SAMPLE SILHOUETTE SCORES ---")

    # Returns an array containing the silhouette score for each data point
    si_samples = cm.SI(multi_output=True)
    print(f"Silhouette Scores per sample:\n{si_samples}")

    # ==============================================================================
    # SCENARIO 3: Edge Case with 1 Cluster
    # ==============================================================================
    print("\n--- 3. EDGE CASE (1 CLUSTER) EXAMPLE ---")

    y_pred_single = np.array([0, 0, 0, 0, 0, 0])
    cm_single = ClusteringMetric(X=X_data, y_pred=y_pred_single)

    # Returns the penalty finite_value (-1.0) instead of crashing
    si_safe = cm_single.SI(force_finite=True, finite_value=-1.0)
    print(f"SI with 1 cluster (Safe Mode): {si_safe}")
