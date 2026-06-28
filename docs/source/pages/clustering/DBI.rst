DBI - Davies-Bouldin Index
==========================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Davies-Bouldin Index (DBI)** is an internal evaluation metric for clustering algorithms :cite:`davies1979cluster`. It is defined as the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances.

Intuitively, DBI evaluates how well the clustering has separated the data. It answers the question: *"For each cluster, how distinct is it from its closest neighboring cluster?"* Clusters that are far apart and highly compact will result in a lower DBI score. A smaller DBI indicates a better clustering partition.

.. math::

    \text{DBI} = \frac{1}{K} \sum_{k=1}^{K} \max_{j \neq k} \left( \frac{\Delta_k + \Delta_j}{\delta(c_k, c_j)} \right)

Where:

* :math:`K` is the total number of clusters.
* :math:`c_k` and :math:`c_j` are the centroids of clusters :math:`k` and :math:`j` respectively.
* :math:`\Delta_k` is the intra-cluster dispersion (average distance of all elements in cluster :math:`k` to its centroid :math:`c_k`).
* :math:`\delta(c_k, c_j)` is the inter-cluster separation (Euclidean distance between centroids :math:`c_k` and :math:`c_j`).

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The Davies-Bouldin index requires comparing at least two distinct clusters to calculate the inter-cluster separation :math:`\delta(c_k, c_j)`. It is mathematically undefined when there is only one cluster (:math:`K = 1`).

* **force_finite (bool):** If ``True``, the function will catch the undefined mathematical operation and return a safe, finite number instead of raising a ``ValueError``. Default is ``True``.
* **finite_value (float):** The specific fallback value returned when ``force_finite=True`` and the clustering has only 1 cluster. Because a smaller score is better for DBI, the default fallback is a large penalty value (``1e10``).

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better. A score of 0 implies perfectly compact clusters that are infinitely far apart).
* **Worst possible score:** ``+inf`` (or the defined penalty ``finite_value``).
* **Range:** ``[0.0, +inf)``
* **References:** `Scikit-Learn Davies-Bouldin <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 12,13,23,26

    from permetrics.clustering import ClusteringMetric
    import numpy as np

    # ==============================================================================
    # SCENARIO 1: Normal Clustering Evaluation
    # ==============================================================================
    print("--- 1. BASIC DAVIES-BOULDIN INDEX EXAMPLE ---")

    X_data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    y_pred_labels = np.array([0, 0, 0, 1, 1, 1])

    cm = ClusteringMetric(X=X_data, y_pred=y_pred_labels)
    dbi_score = cm.DBI()
    print(f"Davies-Bouldin Index: {dbi_score}")

    # ==============================================================================
    # SCENARIO 2: Edge Case with 1 Cluster (Demonstrating force_finite)
    # ==============================================================================
    print("\n--- 2. EDGE CASE (1 CLUSTER) EXAMPLE ---")

    # All data points are predicted to be in the same single cluster (label 0)
    y_pred_single = np.array([0, 0, 0, 0, 0, 0])
    cm_single = ClusteringMetric(X=X_data, y_pred=y_pred_single)

    # Returns the penalty finite_value (1e10) instead of crashing
    dbi_safe = cm_single.DBI(force_finite=True, finite_value=1e10)
    print(f"DBI with 1 cluster (Safe Mode): {dbi_safe}")
