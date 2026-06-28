DI - Dunn Index
===============

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Dunn Index (DI)** :cite:`dunn1974well` is an internal clustering validation metric designed to identify sets of clusters that are compact and well-separated. It is defined as the quotient of the minimal distance between points of different clusters and the largest within-cluster distance.

Intuitively, it answers the question: *"What is the ratio of the shortest distance between any two distinct clusters to the size of the largest cluster?"* A higher Dunn Index indicates better clustering, meaning clusters are compact (small denominator) and far apart from each other (large numerator).

.. math::

    \text{DI} = \frac{d_{\min}}{d_{\max}}

Where the standard definitions are:

* :math:`d_{\min} = \min_{k \neq k'} \left( \min_{i \in I_k, j \in I_{k'}} ||M_i^{\{k\}} - M_j^{\{k'\}}|| \right)` (The smallest distance between any two points belonging to different clusters).
* :math:`d_{\max} = \max_{1 \le k \le K} \left( \max_{i, j \in I_k} ||M_i^{\{k\}} - M_j^{\{k\}}|| \right)` (The maximum diameter among all clusters, i.e., the largest distance between two points in the same cluster).

-------------------------------------------------------------------------------

Algorithmic Variations
----------------------

Calculating the exact minimum distance between all pairs of points across different clusters (:math:`d_{\min}`) has a high computational complexity of :math:`O(N^2)`. To optimize performance on larger datasets, the function provides a modified version:

* **use_modified (bool):** * If ``False``, computes the exact, standard Dunn Index using all point-to-point inter-cluster distances.
    * If ``True`` (default), computes a modified, faster version where :math:`d_{\min}` is calculated as the minimum distance between the data points of one cluster and the **centroid** of another cluster. This drastically reduces computation time while maintaining the core logic of measuring cluster separation.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The Dunn index requires at least two clusters to measure inter-cluster separation. It is mathematically undefined when there is only one cluster (:math:`K = 1`).

* **force_finite (bool):** If ``True``, the function catches the undefined operation and returns a safe, finite number instead of raising a ``ValueError``. Default is ``True``.
* **finite_value (float):** The specific fallback value returned when ``force_finite=True`` and the clustering has only 1 cluster. Since a larger score is better for DI, the default fallback is a penalty value of ``0.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``+inf`` (Larger value is better, indicating well-separated and highly compact clusters).
* **Worst possible score:** ``0.0``
* **Range:** ``[0.0, +inf)``

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 12,13,23,26

    from permetrics.clustering import ClusteringMetric
    import numpy as np

    # ==============================================================================
    # SCENARIO 1: Normal Evaluation (Modified/Optimized Version)
    # ==============================================================================
    print("--- 1. MODIFIED DUNN INDEX EXAMPLE ---")

    X_data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    y_pred_labels = np.array([0, 0, 0, 1, 1, 1])

    cm = ClusteringMetric(X=X_data, y_pred=y_pred_labels)
    # Calculates DI with use_modified=True by default for better performance
    di_modified = cm.DI()
    print(f"Dunn Index (Modified): {di_modified}")

    # ==============================================================================
    # SCENARIO 2: Strict Standard Evaluation (Mathematical Exactness)
    # ==============================================================================
    print("\n--- 2. STANDARD DUNN INDEX EXAMPLE ---")

    # Forces the exact O(N^2) calculation using point-to-point distances
    di_standard = cm.DI(use_modified=False)
    print(f"Dunn Index (Standard): {di_standard}")

    # ==============================================================================
    # SCENARIO 3: Edge Case with 1 Cluster
    # ==============================================================================
    print("\n--- 3. EDGE CASE (1 CLUSTER) EXAMPLE ---")

    y_pred_single = np.array([0, 0, 0, 0, 0, 0])
    cm_single = ClusteringMetric(X=X_data, y_pred=y_pred_single)

    # Returns the penalty finite_value (0.0)
    di_safe = cm_single.DI(force_finite=True, finite_value=0.0)
    print(f"DI with 1 cluster (Safe Mode): {di_safe}")


.. toctree::
   :maxdepth: 3
   :caption: Dunn Index (DI)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


The Dunn Index, which is a measure used to evaluate the performance of clustering algorithms. The Dunn Index aims to quantify the compactness and separation
between clusters in a clustering solution. It helps assess the quality of the clustering by considering both the distance between points within the same cluster (intra-cluster distance) and the distance between points in different clusters (inter-cluster distance).


.. image:: /_static/images/DI.png


A higher Dunn Index value indicates better clustering quality – it suggests that the clusters are well separated from each other while being compact internally. Conversely, a lower Dunn Index value may indicate that the clusters are too spread out or not well separated.

However, like any clustering evaluation metric, the Dunn Index has its limitations and should be used in conjunction with other metrics and domain knowledge. It's worth noting that the choice of clustering algorithm, distance metric, and dataset characteristics can influence the interpretation of the Dunn Index.


Example:

.. code-block:: python

	import numpy as np
	from permetrics import ClusteringMetric

	## For integer labels or categorical labels
	data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
	y_pred = np.array([0, 0, 1, 1, 1])

	cm = ClusteringMetric(X=data, y_pred=y_pred)

	print(cm.dunn_index())
	print(cm.DI())
