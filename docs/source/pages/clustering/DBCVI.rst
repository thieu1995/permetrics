DBCVI - Density-Based Clustering Validation Index
=================================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Density-Based Clustering Validation Index (DBCVI)** index is an internal evaluation metric specifically designed for density-based clustering algorithms (such as DBSCAN or OPTICS). Unlike traditional metrics (e.g., Silhouette, Davies-Bouldin) which assume spherical clusters and rely on centroids, DBCVI evaluates arbitrary-shaped clusters by analyzing the density connectedness of the data points using Minimum Spanning Trees (MST).

Intuitively, DBCVI answers the question: *"Are the clusters truly dense continuous regions separated by areas of lower density, regardless of their geometric shape?"* -------------------------------------------------------------------------------

Mathematical Formulation
----------------------

DBCVI operates entirely without cluster centroids. Instead, it relies on the concept of **Mutual Reachability Distance (MRD)** to construct a shape-aware density graph.

**1. All-Points Core Distance**
To estimate the local density of an object :math:`o` within its cluster :math:`C_i`, DBCVI calculates the all-points core distance:

.. math::

    a_{pts\_coredist}(o) = \left( \frac{\sum_{k=2}^{n_i} \left( KNN(o, k) \right)^d}{n_i - 1} \right)^{1/d}

Where :math:`d` is the dimensionality of the feature space, and the summation runs over all other points in the same cluster.

**2. Mutual Reachability Distance (MRD)**
The distance between two objects considering their density properties is defined as:

.. math::

    d_{mreach}(o_i, o_j) = \max \left( a_{pts\_coredist}(o_i), a_{pts\_coredist}(o_j), d(o_i, o_j) \right)

**3. Density Sparseness and Separation**
A Minimum Spanning Tree (:math:`\text{MST}_{MRD}`) is constructed for each cluster using the MRD.

* **Density Sparseness (DSC):** The maximum edge weight of the internal edges in the :math:`\text{MST}_{MRD}` of cluster :math:`C_i`. It represents the lowest density (sparsest) region *inside* the cluster.
* **Density Separation (DSPC):** The minimum MRD between the internal nodes of cluster :math:`C_i` and cluster :math:`C_j`. It represents the highest density region *between* the clusters.

**4. Cluster Validity**
The validity of a single cluster :math:`C_i` is bounded in ``[-1, 1]``:

.. math::

    V_C(C_i) = \frac{\min_{j \neq i} (DSPC(C_i, C_j)) - DSC(C_i)}{\max \left( \min_{j \neq i} (DSPC(C_i, C_j)), DSC(C_i) \right)}

**5. Global DBCVI Index**
The final score is the weighted average of all cluster validities, heavily penalizing unclustered noise objects (:math:`|O|` includes noise):

.. math::

    \text{DBCVI}(C) = \sum_{i=1}^{l} \frac{|C_i|}{|O|} V_C(C_i)

-------------------------------------------------------------------------------

Handling Edge Cases & API
---------------------------

* **Noise Objects:** Standard density algorithms often label outliers as noise (typically ``-1``). DBCVI naturally handles noise by excluding noise points from the MST construction, but penalizing the final score by scaling it down by the ratio of clustered points to total points.
* **force_finite (bool):** If ``True``, catches mathematical edge cases (e.g., fewer than 2 valid clusters) and returns a safe fallback value. Default is ``True``.
* **finite_value (float):** The fallback value returned when calculation fails. Default is ``0.0``.
* **return_type (str):** Controls the output format of the evaluation.
    * ``"global"`` (Default): Returns a single ``float`` representing the weighted overall DBCV score.
    * ``"per-cluster"``: Returns a ``dict`` mapping each valid cluster label to its individual validity score. Useful for debugging which specific cluster is underperforming.
    * ``"both"``: Returns a tuple containing ``(global_score, per_cluster_dict)``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates perfectly dense clusters perfectly separated by empty space).
* **Worst possible score:** ``-1.0`` (Indicates that the density between clusters is higher than the density within clusters).
* **Shape Invariance:** Effectively evaluates non-convex, elongated, or arbitrarily shaped clusters.
* **References:** `Moulavi, D., Jaskowiak, P. A., Campello, R. J., Zimek, A., & Sander, J. (2014, April). Density-based clustering validation. In Proceedings of the 2014 SIAM international conference on data mining (pp. 839-847). Society for Industrial and Applied Mathematics. <https://doi.org/10.1137/1.9781611973440.96>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 12,15,19,24

    from permetrics.clustering import ClusteringMetric
    import numpy as np

    # ==============================================================================
    # SCENARIO: Density-based evaluation with noise handling
    # ==============================================================================
    # Assume a dataset where DBSCAN grouped 90 points into 2 clusters,
    # and labeled 10 outlier points as noise (-1)
    X = np.random.rand(100, 2)
    y_pred = np.array([0]*45 + [1]*45 + [-1]*10)

    cm = ClusteringMetric(X=X, y_pred=y_pred)

    # 1. Global Score (Default)
    dbcv_global = cm.DBCV()
    print(f"Global DBCV: {dbcv_global:.4f}")

    # 2. Per-cluster Validity Scores
    dbcv_clusters = cm.DBCV(return_type="per-cluster")
    print(f"Per-cluster Scores: {dbcv_clusters}")
    # Output example: {0: 0.8123, 1: 0.7501}

    # 3. Extracting Both Simultaneously
    gb_score, cluster_dict = cm.DBCV(return_type="both")
