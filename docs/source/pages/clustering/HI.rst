HI - Hartigan Index
===================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Hartigan Index (HI)** is an internal clustering evaluation metric. It assesses the clustering quality by calculating the ratio of the within-cluster sum of squares to the sum of squares between clusters, specifically focusing on the relative dispersion of each cluster compared to its nearest neighbor.

Intuitively, HI answers the question: *"Does the internal compactness of a cluster justify its existence compared to the next closest cluster?"* A lower HI value indicates a better clustering partition, as it implies that the intra-cluster dispersion is small relative to the distance to the nearest competing cluster.

.. math::

    \text{HI} = \sum_{k=1}^{K} \left( \frac{\sum_{x_i \in C_k} ||x_i - c_k||^2}{\sum_{x_i \in C_k} ||x_i - c_{\text{nearest}}||^2} \right)

Where:

* :math:`K` is the total number of clusters.
* :math:`c_k` is the centroid of cluster :math:`k`.
* :math:`c_{\text{nearest}}` is the centroid of the cluster closest to cluster :math:`k`.
* The numerator is the within-cluster dispersion (SSE) of cluster :math:`k`.
* The denominator is the dispersion of cluster :math:`k` relative to the nearest neighboring cluster.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The Hartigan Index involves comparing clusters and their neighbors. It is mathematically undefined when there is only one cluster (:math:`K = 1`), as there are no "nearest neighbors" to compare against.

* **force_finite (bool):** If ``True``, the function catches the undefined operation and returns a safe, finite number instead of raising a ``ValueError``. Default is ``True``.
* **finite_value (float):** The fallback value returned when ``force_finite=True`` and the clustering has only 1 cluster. Since a smaller score is better for HI, the default fallback is a large penalty value (``1e10``).

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better).
* **Worst possible score:** ``+inf`` (or the defined penalty ``finite_value``).
* **Range:** ``[0.0, +inf)``

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 12,13,22,25

    from permetrics.clustering import ClusteringMetric
    import numpy as np

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC HARTIGAN INDEX EXAMPLE ---")

    X_data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    y_pred_labels = np.array([0, 0, 0, 1, 1, 1])

    cm = ClusteringMetric(X=X_data, y_pred=y_pred_labels)
    hi_score = cm.HI()
    print(f"Hartigan Index: {hi_score}")

    # ==============================================================================
    # SCENARIO 2: Edge Case with 1 Cluster
    # ==============================================================================
    print("\n--- 2. EDGE CASE (1 CLUSTER) EXAMPLE ---")

    y_pred_single = np.array([0, 0, 0, 0, 0, 0])
    cm_single = ClusteringMetric(X=X_data, y_pred=y_pred_single)

    # Returns the penalty finite_value (1e10)
    hi_safe = cm_single.HI(force_finite=True, finite_value=1e10)
    print(f"HI with 1 cluster (Safe Mode): {hi_safe}")
