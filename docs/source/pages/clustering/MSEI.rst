MSEI - Mean Squared Error Index
===============================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Mean Squared Error Index (MSEI)** is a direct derivative of the Sum of Squared Error Index (SSEI). While SSEI computes the *total* within-cluster dispersion, MSEI calculates the *average* dispersion per data point.

Intuitively, it answers the question: *"On average, what is the squared distance from any given data point to its assigned cluster centroid?"* A lower MSEI indicates that the clusters are highly compact and the data points are tightly grouped around their respective centers.

.. math::

    \text{MSEI} = \frac{1}{N} \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - c_k||^2

Where:

* :math:`N` is the total number of data points (samples).
* :math:`K` is the total number of clusters.
* :math:`C_k` is the set of data points assigned to the :math:`k`-th cluster.
* :math:`c_k` is the centroid (mean) of cluster :math:`C_k`.
* :math:`x_i` is a data point belonging to cluster :math:`C_k`.

-------------------------------------------------------------------------------

Algorithmic Variations (Performance Note)
-----------------------------------------

Similar to the SSEI metric, this implementation calculates the mean dispersion using a highly optimized, vectorized approach. By mapping the precomputed centroids directly to the samples via ``centers[y_pred]``, it completely avoids inefficient nested loops and executes in :math:`O(N)` time, making it exceptionally fast and memory-safe for large datasets.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better. A score of 0 indicates that every data point lies perfectly on top of its cluster centroid).
* **Worst possible score:** No strict upper bound. 
* **Range:** ``[0.0, +inf)``

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 14,15,24

    from permetrics.clustering import ClusteringMetric
    import numpy as np

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC MEAN SQUARED ERROR INDEX EXAMPLE ---")

    # Features (X) and predicted cluster labels (y_pred)
    X_data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    y_pred_labels = np.array([0, 0, 0, 1, 1, 1])
    
    # Initialize the metric object
    cm = ClusteringMetric(X=X_data, y_pred=y_pred_labels)
    msei_score = cm.MSEI()
    print(f"Mean Squared Error Index: {msei_score}")

    # ==============================================================================
    # SCENARIO 2: Using the static method directly
    # ==============================================================================
    print("\n--- 2. STATIC METHOD USAGE ---")

    # Bypass object instantiation if you only need a single calculation
    msei_static = ClusteringMetric.calculate_mean_squared_error_index(X=X_data, y_pred=y_pred_labels)
    print(f"Mean Squared Error Index (Static): {msei_static}")
