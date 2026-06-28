LSRI - Log SS Ratio Index
=========================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Log SS Ratio Index (LSRI)** is an internal clustering evaluation metric. It computes the natural logarithm of the ratio between the between-group dispersion (BGSS) and the within-group dispersion (WGSS).

Intuitively, LSRI evaluates clustering quality by comparing how well the clusters are separated against how compact they are. A higher LSRI indicates that clusters are well-separated (high BGSS) and highly compact (low WGSS). The logarithmic scale helps to smoothly manage exceptionally large or small dispersion ratios.

.. math::

    \text{LSRI} = \log \left( \frac{\text{BGSS}}{\text{WGSS}} \right)

Where:

* :math:`\text{BGSS}` is the Between-Group Sum of Squares (the sum of the squared distances between the cluster centroids and the overall data centroid, weighted by cluster size).
* :math:`\text{WGSS}` is the Within-Group Sum of Squares (the pooled within-cluster dispersion).

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The calculation of LSRI involves division by :math:`\text{WGSS}` and evaluating a natural logarithm, which can trigger mathematical exceptions in edge cases:

1. **Single Cluster:** If there is only 1 cluster (:math:`K = 1`), there is no between-group dispersion (:math:`\text{BGSS} = 0`). The ratio evaluates to 0, making :math:`\log(0)` mathematically undefined.
2. **Zero Variance:** If all data points within every cluster are perfectly identical to their respective centroids, :math:`\text{WGSS} = 0`, causing a zero-division error.

* **force_finite (bool):** If ``True``, the function catches these undefined operations and returns a safe, finite number instead of raising a ``ValueError`` or ``ZeroDivisionError``. Default is ``True``.
* **finite_value (float):** The specific fallback value returned when ``force_finite=True``. Since a larger score is better for LSRI, the default fallback is a large negative penalty value (``-1e10``).

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``+inf`` (Larger value is better).
* **Worst possible score:** ``-inf`` (or the defined penalty ``finite_value``).
* **Range:** ``(-inf, +inf)``
* **References:** `Hartigan, J. A. (1975). Clustering algorithms. New York: Wiley. <https://dl.acm.org/doi/abs/10.5555/540298>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 12,13,22,25

    from permetrics.clustering import ClusteringMetric
    import numpy as np

    # ==============================================================================
    # SCENARIO 1: Normal Clustering Evaluation
    # ==============================================================================
    print("--- 1. BASIC LOG SS RATIO INDEX EXAMPLE ---")

    X_data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    y_pred_labels = np.array([0, 0, 0, 1, 1, 1])
    
    cm = ClusteringMetric(X=X_data, y_pred=y_pred_labels)
    lsri_score = cm.LSRI()
    print(f"Log SS Ratio Index: {lsri_score}")

    # ==============================================================================
    # SCENARIO 2: Edge Case with 1 Cluster
    # ==============================================================================
    print("\n--- 2. EDGE CASE (1 CLUSTER) EXAMPLE ---")

    y_pred_single = np.array([0, 0, 0, 0, 0, 0])
    cm_single = ClusteringMetric(X=X_data, y_pred=y_pred_single)

    # Returns the penalty finite_value (-1e10) instead of crashing
    lsri_safe = cm_single.LSRI(force_finite=True, finite_value=-1e10)
    print(f"LSRI with 1 cluster (Safe Mode): {lsri_safe}")
