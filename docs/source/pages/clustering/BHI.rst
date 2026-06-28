BHI - Ball-Hall Index
=====================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Ball-Hall Index (BHI)** is an internal clustering evaluation metric that measures the mean of the within-cluster dispersions. Introduced in 1965, it is based on the Sum of Squared Errors (SSE) within each cluster.

Intuitively, the Ball-Hall Index evaluates how compact the clusters are. It answers the question: *"On average, how far are the data points in a cluster from their respective centroid?"* A smaller value indicates denser, more compact clusters.

.. math::

    \text{BHI} = \frac{1}{K} \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \bar{x}_k||^2

Where:

* :math:`K` is the total number of clusters.
* :math:`C_k` is the set of data points assigned to the :math:`k`-th cluster.
* :math:`\bar{x}_k` is the centroid (mean) of cluster :math:`C_k`.
* :math:`x_i` is a data point belonging to cluster :math:`C_k`.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better. A score of 0 indicates that all data points perfectly overlap with their cluster centroids).
* **Worst possible score:** No strictly defined upper bound (higher values mean higher within-cluster dispersion).
* **Range:** ``[0.0, +inf)``
* **References:** `Ball, G.H. and Hall, D.J. (1965) ISODATA: A Novel Method of Data Analysis and Pattern Classification. <https://apps.dtic.mil/sti/pdfs/AD0699616.pdf>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 14,17,26,27

    from permetrics.clustering import ClusteringMetric
    import numpy as np

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation (Internal Metric requires X and y_pred)
    # ==============================================================================
    print("--- 1. BASIC BALL-HALL INDEX EXAMPLE ---")

    # Features (X) and predicted cluster labels (y_pred)
    X_data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    y_pred_labels = np.array([0, 0, 0, 1, 1, 1])

    # Initialize the metric object
    cm = ClusteringMetric(X=X_data, y_pred=y_pred_labels)

    # Calculate the Ball-Hall Index
    bhi_score = cm.BHI()
    print(f"Ball-Hall Index: {bhi_score}")

    # ==============================================================================
    # SCENARIO 2: Using the static method directly
    # ==============================================================================
    print("\n--- 2. STATIC METHOD USAGE ---")

    # If you prefer to bypass object instantiation
    bhi_static = ClusteringMetric.ball_hall_index(X=X_data, y_pred=y_pred_labels)
    print(f"Ball-Hall Index (Static): {bhi_static}")
