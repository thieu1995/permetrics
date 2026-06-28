SSEI - Sum of Squared Error Index
=================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Sum of Squared Error Index (SSEI)** :cite:`edwards1965method`, commonly referred to in clustering literature as the **Within-Cluster Sum of Squares (WCSS)** or **Trace_W**, is a fundamental internal evaluation metric. It measures the total dispersion of data points within their assigned clusters.

Intuitively, SSEI evaluates the absolute compactness of the clustering partition. It answers the question: *"What is the total sum of squared distances from every data point to its respective cluster centroid?"* A lower SSEI value indicates that the data points are tightly packed around their centroids.

.. math::

    \text{SSEI} = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - c_k||^2

Where:

* :math:`K` is the total number of clusters.
* :math:`C_k` is the set of data points assigned to the :math:`k`-th cluster.
* :math:`c_k` is the centroid (mean) of cluster :math:`C_k`.
* :math:`x_i` is a data point belonging to cluster :math:`C_k`.

-------------------------------------------------------------------------------

Algorithmic Variations (Performance Note)
-----------------------------------------

This metric calculates the core internal dispersion used as a foundation for many other complex indices (like Calinski-Harabasz or Ball-Hall).

In this implementation, the calculation is highly vectorized. Instead of using nested loops to iterate through clusters and points, it maps the computed centroids directly to the samples using the predicted labels (``centers[y_pred]``). This reduces the computational overhead significantly, allowing it to run efficiently in :math:`O(N)` time even on large datasets.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better. A score of 0 indicates that every data point lies exactly on top of its cluster centroid).
* **Worst possible score:** No strict upper bound. The worst case is when :math:`K=1`, where the SSEI equals the Total Sum of Squares (TSS) of the entire dataset.
* **Range:** ``[0.0, +inf)``

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 14,15,24,27

    from permetrics.clustering import ClusteringMetric
    import numpy as np

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC SUM OF SQUARED ERROR INDEX EXAMPLE ---")

    # Features (X) and predicted cluster labels (y_pred)
    X_data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    y_pred_labels = np.array([0, 0, 0, 1, 1, 1])

    # Initialize the metric object
    cm = ClusteringMetric(X=X_data, y_pred=y_pred_labels)
    ssei_score = cm.SSEI()
    print(f"Sum of Squared Error Index: {ssei_score}")

    # ==============================================================================
    # SCENARIO 2: Single Cluster Evaluation (Equals Total Sum of Squares)
    # ==============================================================================
    print("\n--- 2. SINGLE CLUSTER (TSS) EXAMPLE ---")

    y_pred_single = np.array([0, 0, 0, 0, 0, 0])
    cm_single = ClusteringMetric(X=X_data, y_pred=y_pred_single)

    # Valid calculation; returns the dispersion of the entire dataset
    ssei_single = cm_single.SSEI()
    print(f"SSEI with 1 cluster (TSS): {ssei_single}")
