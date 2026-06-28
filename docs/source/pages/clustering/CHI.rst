CHI - Calinski-Harabasz Index
=============================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Calinski-Harabasz Index (CHI)** also known as the **Variance Ratio Criterion**, is an internal clustering evaluation metric :cite:`calinski1974dendrite` . It computes the ratio of the sum of between-cluster dispersion to within-cluster dispersion for all clusters.

Intuitively, CHI evaluates the validity of a clustering based on the average between- and within-cluster sum of squares. It answers the question: *"How well-separated are the clusters relative to how compact they are?"* A higher score implies that clusters are dense (low within-cluster variance) and well-separated (high between-cluster variance).

.. math::

    \text{CHI} = \frac{\text{Tr}(B_K)}{\text{Tr}(W_K)} \times \frac{N - K}{K - 1}

Where:

* :math:`N` is the total number of data points (samples).
* :math:`K` is the number of clusters.
* :math:`\text{Tr}(B_K)` is the trace of the between-group dispersion matrix.
* :math:`\text{Tr}(W_K)` is the trace of the within-cluster dispersion matrix.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The Calinski-Harabasz index is mathematically undefined when there is only one cluster (:math:`K = 1`), as the denominator :math:`(K - 1)` becomes zero. The function provides parameters to safely handle this scenario:

* **force_finite (bool):** If ``True``, the function will catch the undefined mathematical operation and return a safe, finite number instead of raising an exception. Default is ``True``.
* **finite_value (float):** The specific fallback value returned when ``force_finite=True`` and the clustering has only 1 cluster. Default is ``0.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** No strict upper bound (Higher value is better).
* **Worst possible score:** ``0.0``
* **Range:** ``[0.0, +inf)``
* **Notes:**  This metric in scikit-learn library is wrong in calculate the intra_disp variable (WGSS) `Scikit-Learn Calinski-Harabasz <https://github.com/scikit-learn/scikit-learn/blob/7f9bad99d/sklearn/metrics/cluster/_unsupervised.py#L351C1-L351C1>`_

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
    print("--- 1. BASIC CALINSKI-HARABASZ INDEX EXAMPLE ---")

    X_data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    y_pred_labels = np.array([0, 0, 0, 1, 1, 1])

    cm = ClusteringMetric(X=X_data, y_pred=y_pred_labels)
    chi_score = cm.CHI()
    print(f"Calinski-Harabasz Index: {chi_score}")

    # ==============================================================================
    # SCENARIO 2: Edge Case with 1 Cluster (Demonstrating force_finite)
    # ==============================================================================
    print("\n--- 2. EDGE CASE (1 CLUSTER) EXAMPLE ---")

    # All data points are predicted to be in the same single cluster (label 0)
    y_pred_single = np.array([0, 0, 0, 0, 0, 0])
    cm_single = ClusteringMetric(X=X_data, y_pred=y_pred_single)

    # Returns the finite_value (0.0) instead of crashing
    chi_safe = cm_single.CHI(force_finite=True, finite_value=0.0)
    print(f"CHI with 1 cluster (Safe Mode): {chi_safe}")
