DHI - Duda-Hart Index
=====================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Duda-Hart Index (DHI)** (adapted formulation) is an internal clustering validation metric. It assesses the quality of a clustering partition by evaluating the ratio of the average intra-cluster pairwise distances to the average inter-cluster pairwise distances.

Intuitively, DHI answers the question: *"How close are points within the same cluster compared to their distance to points in all other clusters?"* A smaller DHI score implies an optimal clustering structure, where clusters are highly cohesive (small numerator) and well-isolated from one another (large denominator).

.. math::

    \text{DHI} = \frac{\sum_{k=1}^{K} D_{\text{intra}}(C_k)}{\sum_{k=1}^{K} D_{\text{inter}}(C_k)}

Where:

* :math:`K` is the total number of clusters.
* :math:`D_{\text{intra}}(C_k)` is the mean Euclidean distance between all pairs of points within cluster :math:`C_k`.
* :math:`D_{\text{inter}}(C_k)` is the mean Euclidean distance between points in cluster :math:`C_k` and points in all other distinct clusters.

-------------------------------------------------------------------------------

Algorithmic Variations (Memory Optimization)
--------------------------------------------

Calculating pairwise distances for the Duda-Hart index typically requires instantiating an :math:`N \times N` distance matrix, which causes severe Out-Of-Memory (OOM) errors on large datasets (e.g., :math:`N > 40,000` consuming 10GB+ RAM).

This implementation employs a highly optimized **chunk-based matrix multiplication** strategy:

* **chunk_size (int):** Computes distances in batches (default: ``2000``). This algorithm tightly caps RAM consumption to a safe minimum (~1.5GB for 100K samples) while mathematically guaranteeing exact parity with the standard :math:`O(N^2)` memory approach.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The calculation of DHI requires comparing distances between at least two distinct clusters. If all data points are assigned to a single cluster (:math:`K = 1`), the inter-cluster distance (denominator) is zero, rendering the metric mathematically undefined.

* **force_finite (bool):** If ``True``, the function catches the undefined operation and returns a safe, finite number instead of raising a ``ValueError`` or ``ZeroDivisionError``. Default is ``True``.
* **finite_value (float):** The fallback value returned when ``force_finite=True`` and the clustering fails edge-case checks. Since a smaller score is better for DHI, the default fallback is a massive penalty value (``1e10``).

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better. A score approaching 0 means clusters are perfectly dense dots separated by vast distances).
* **Worst possible score:** ``+inf`` (or the defined penalty ``finite_value``).
* **Range:** ``[0.0, +inf)``
* **References:** `Duda, R. O., & Hart, P. E. (1973). Pattern classification and scene analysis. John Wiley & Sons. <https://www.wiley.com/en-us/Pattern+Classification%2C+2nd+Edition-p-9780471056690>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 12,14,23,32,35

    from permetrics.clustering import ClusteringMetric
    import numpy as np

    # ==============================================================================
    # SCENARIO 1: Normal Evaluation (Memory-Optimized)
    # ==============================================================================
    print("--- 1. BASIC DUDA-HART INDEX EXAMPLE ---")

    X_data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    y_pred_labels = np.array([0, 0, 0, 1, 1, 1])

    cm = ClusteringMetric(X=X_data, y_pred=y_pred_labels)
    # Calculates DHI (implicitly uses safe chunk_size=2000)
    dhi_score = cm.DHI()
    print(f"Duda-Hart Index: {dhi_score}")

    # ==============================================================================
    # SCENARIO 2: Adjusting Chunk Size for Hardware Restraints
    # ==============================================================================
    print("\n--- 2. HARDWARE OPTIMIZATION EXAMPLE ---")

    # Lower the chunk_size if running on extremely constrained RAM environments
    dhi_constrained = cm.DHI(chunk_size=1000)
    print(f"Duda-Hart Index (Lower RAM cap): {dhi_constrained}")

    # ==============================================================================
    # SCENARIO 3: Edge Case with 1 Cluster
    # ==============================================================================
    print("\n--- 3. EDGE CASE (1 CLUSTER) EXAMPLE ---")

    y_pred_single = np.array([0, 0, 0, 0, 0, 0])
    cm_single = ClusteringMetric(X=X_data, y_pred=y_pred_single)

    # Returns the penalty finite_value (1e10) instead of crashing
    dhi_safe = cm_single.DHI(force_finite=True, finite_value=1e10)
    print(f"DHI with 1 cluster (Safe Mode): {dhi_safe}")
