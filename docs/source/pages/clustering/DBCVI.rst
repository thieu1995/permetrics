DBCVI - Density-Based Clustering Validation Index
=================================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Density-based Clustering Validation Index (DBCVI)** :cite:`halkidi2001clustering` is an internal evaluation metric designed specifically for clusters of arbitrary shapes and varying densities. Unlike variance-based metrics (like SSEI), DBCVI evaluates cluster quality by measuring the ratio of the average intra-cluster density to the inter-cluster density.

Intuitively, DBCVI answers: *"Is the density within my clusters significantly higher than the density in the space between the clusters?"* A lower score indicates that the clusters are both well-separated and internally dense, effectively validating structures that traditional centroid-based metrics would fail to capture.

.. math::

    \text{DBCVI} = \frac{1}{K} \sum_{k=1}^{K} \left( \frac{\max_{j \neq k} \text{density}(C_k, C_j)}{\text{density}(C_k)} \right)

Where:

* :math:`K` is the total number of clusters.
* :math:`\text{density}(C_k)` measures the internal cohesion (e.g., average distance to centroid).
* :math:`\text{density}(C_k, C_j)` measures the separation between clusters :math:`C_k` and :math:`C_j`.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

DBCVI requires the existence of inter-cluster separation. If all points belong to a single cluster (:math:`K = 1`), the density between clusters is undefined.

* **force_finite (bool):** If ``True``, the function catches the undefined operation and returns a safe, finite number. Default is ``True``.
* **finite_value (float):** The fallback value when :math:`K = 1`. Default is ``1.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better, indicating clusters are highly dense and well-separated).
* **Worst possible score:** No strict upper bound.
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
    print("--- 1. BASIC DENSITY-BASED CLUSTERING VALIDATION INDEX EXAMPLE ---")

    X_data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    y_pred_labels = np.array([0, 0, 0, 1, 1, 1])

    cm = ClusteringMetric(X=X_data, y_pred=y_pred_labels)
    dbcvi_score = cm.DBCVI()
    print(f"Density-Based Clustering Validation Index: {dbcvi_score}")

    # ==============================================================================
    # SCENARIO 2: Edge Case with 1 Cluster
    # ==============================================================================
    print("\n--- 2. EDGE CASE (1 CLUSTER) EXAMPLE ---")

    y_pred_single = np.array([0, 0, 0, 0, 0, 0])
    cm_single = ClusteringMetric(X=X_data, y_pred=y_pred_single)

    # Returns the penalty finite_value (1.0) instead of crashing
    dbcvi_safe = cm_single.DBCVI(force_finite=True, finite_value=1.0)
    print(f"DBCVI with 1 cluster (Safe Mode): {dbcvi_safe}")
