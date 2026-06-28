XBI - Xie-Beni Index
====================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Xie-Beni Index (XBI)** :cite:`xie1991validity` is an internal clustering validation metric that measures the ratio of the total within-cluster variance (compactness) to the minimum squared distance between cluster centroids (separation).

Originally introduced for fuzzy clustering, it is widely adapted for hard clustering evaluations. Intuitively, it answers the question: *"How compact are the clusters relative to the distance between the two closest clusters?"* A smaller XBI value indicates a better clustering partition, implying that clusters are highly compact and well-separated.

.. math::

    \text{XBI} = \frac{\frac{1}{N} \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - c_k||^2}{\min_{j \neq k} ||c_j - c_k||^2}

Where:

* :math:`N` is the total number of data points.
* :math:`K` is the number of clusters.
* :math:`C_k` is the set of data points assigned to the :math:`k`-th cluster.
* :math:`c_k` and :math:`c_j` are the centroids of clusters :math:`k` and :math:`j` respectively.
* The numerator represents the mean squared error (WGSS / N).
* The denominator represents the minimum squared Euclidean distance between any two cluster centroids.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The Xie-Beni index requires calculating the distance between at least two distinct cluster centroids. It is mathematically undefined when there is only one cluster (:math:`K = 1`). 

* **force_finite (bool):** If ``True``, the function catches the undefined operation and returns a safe, finite number instead of raising a ``ValueError``. Default is ``True``.
* **finite_value (float):** The specific fallback value returned when ``force_finite=True`` and the clustering has only 1 cluster. Since smaller is better for XBI, the default fallback is a large penalty value (``1e10``).

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
    :emphasize-lines: 12,13,23,26

    from permetrics.clustering import ClusteringMetric
    import numpy as np

    # ==============================================================================
    # SCENARIO 1: Normal Clustering Evaluation
    # ==============================================================================
    print("--- 1. BASIC XIE-BENI INDEX EXAMPLE ---")

    X_data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    y_pred_labels = np.array([0, 0, 0, 1, 1, 1])
    
    cm = ClusteringMetric(X=X_data, y_pred=y_pred_labels)
    xbi_score = cm.XBI()
    print(f"Xie-Beni Index: {xbi_score}")

    # ==============================================================================
    # SCENARIO 2: Edge Case with 1 Cluster (Demonstrating force_finite)
    # ==============================================================================
    print("\n--- 2. EDGE CASE (1 CLUSTER) EXAMPLE ---")

    # All data points are predicted to be in the same single cluster (label 0)
    y_pred_single = np.array([0, 0, 0, 0, 0, 0])
    cm_single = ClusteringMetric(X=X_data, y_pred=y_pred_single)

    # Returns the penalty finite_value (1e10) instead of crashing
    xbi_safe = cm_single.XBI(force_finite=True, finite_value=1e10)
    print(f"XBI with 1 cluster (Safe Mode): {xbi_safe}")
