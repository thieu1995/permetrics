BRI - Banfeld-Raftery Index
===========================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Banfeld-Raftery Index (BRI)** is an internal clustering evaluation metric derived from maximum-likelihood estimation for model-based clustering :cite:`banfield1993model`. It measures the weighted sum of the logarithms of the within-cluster variances.

Intuitively, BRI evaluates the compactness of the clusters on a logarithmic scale. It answers the question: *"How compact are the clusters when accounting for their varying sizes?"* A smaller (more negative) BRI score indicates denser clusters and a better overall partition.

.. math::

    \text{BRI} = \sum_{k=1}^{K} n_k \ln \left( \frac{\text{Tr}(W_k)}{n_k} \right)

Where:

* :math:`K` is the total number of clusters.
* :math:`n_k` is the number of data points assigned to the :math:`k`-th cluster.
* :math:`W_k` is the within-cluster scatter matrix for cluster :math:`k`.
* :math:`\text{Tr}(W_k)` is the trace of the scatter matrix (the sum of squared Euclidean distances from the points in cluster :math:`k` to their centroid).

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The Banfeld-Raftery index relies on the logarithm of the cluster variance. If any cluster contains only 1 data point (:math:`n_k = 1`), its variance is zero, and :math:`\ln(0)` is mathematically undefined.

* **force_finite (bool):** If ``True``, the function will catch this undefined mathematical operation and return a safe, finite number instead of raising a ``ValueError``. Default is ``True``.
* **finite_value (float):** The specific fallback value returned when ``force_finite=True`` and at least one cluster has only 1 sample. Since a smaller score is better for BRI, the default fallback is a large penalty value (``1e10``).

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** No strict lower bound (Smaller value is better).
* **Worst possible score:** ``+inf`` (or the defined penalty ``finite_value``).
* **Range:** ``(-inf, +inf)``

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
    print("--- 1. BASIC BANFELD-RAFTERY INDEX EXAMPLE ---")

    X_data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    y_pred_labels = np.array([0, 0, 0, 1, 1, 1])

    cm = ClusteringMetric(X=X_data, y_pred=y_pred_labels)
    bri_score = cm.BRI()
    print(f"Banfeld-Raftery Index: {bri_score}")

    # ==============================================================================
    # SCENARIO 2: Edge Case with a Single-Sample Cluster
    # ==============================================================================
    print("\n--- 2. EDGE CASE (SINGLE-SAMPLE CLUSTER) EXAMPLE ---")

    # Cluster label '2' has only 1 sample
    y_pred_single = np.array([0, 0, 0, 1, 1, 2])
    cm_single = ClusteringMetric(X=X_data, y_pred=y_pred_single)

    # Returns the penalty finite_value (1e10) instead of crashing
    bri_safe = cm_single.BRI(force_finite=True, finite_value=1e10)
    print(f"BRI with 1-sample cluster (Safe Mode): {bri_safe}")
