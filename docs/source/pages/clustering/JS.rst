JS - Jaccard Score
==================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Jaccard Score (JS)** (also known as the **Jaccard Index** or **Tanimoto Coefficient**) is an external clustering evaluation metric. It quantifies the similarity between two clustering partitions by measuring the ratio of truly co-clustered sample pairs to the total number of pairs that were grouped together by at least one of the partitions.

Intuitively, JS answers the question: *"Of all the pairs of points that were grouped together in either the ground truth or the model's prediction, what proportion of them were grouped together in both?"* Unlike the Rand Score, the Jaccard Score completely ignores True Negatives (:math:`nn`). This makes it exceptionally useful when analyzing datasets with a large number of clusters, where the vast majority of sample pairs belong to different clusters and would otherwise artificially inflate the similarity score.

.. math::

    \text{JS} = \frac{yy}{yy + yn + ny}

Where across all pairs of distinct data points:

* :math:`yy` (True Positives): Number of pairs placed in the **same** cluster in both the ground truth (:math:`y_{true}`) and the prediction (:math:`y_{pred}`).
* :math:`yn` (False Negatives): Pairs co-clustered in :math:`y_{true}`, but separated in :math:`y_{pred}`.
* :math:`ny` (False Positives): Pairs co-clustered in :math:`y_{pred}`, but separated in :math:`y_{true}`.

-------------------------------------------------------------------------------

Algorithmic Optimizations (Performance Note)
--------------------------------------------

Iterating through all possible pair combinations to evaluate :math:`yy`, :math:`yn`, and :math:`ny` scales quadratically at :math:`O(N^2)`. 

This implementation derives the exact pair totals directly from the algebraic dot products of the **Contingency Matrix** marginals. This reduces the computational complexity to **:math:`O(N)` time**, allowing instantaneous evaluation on massive datasets.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The Jaccard Score involves division by :math:`yy + yn + ny`. If both partitions consist entirely of isolated singletons (every cluster has exactly 1 data point), neither partition groups any points together. The denominator evaluates to zero, causing an undefined mathematical division.

* **force_finite (bool):** If ``True``, catches the zero-division error and returns a safe fallback value instead of raising a ``ZeroDivisionError``. Default is ``True``.
* **finite_value (float):** The fallback value returned when ``force_finite=True`` and the calculation fails. Since the worst possible valid score is 0.0, the default fallback is ``0.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates identical clustering partitions).
* **Worst possible score:** ``0.0`` (The two partitions share zero co-clustered pairs).
* **Permutation Invariance:** The metric is completely invariant to permutations of cluster labels.
* **Symmetry:** Strictly symmetric: :math:`\text{JS}(y_{true}, y_{pred}) = \text{JS}(y_{pred}, y_{true})`.
* **Range:** ``[0.0, 1.0]``
* **References:**

    * `Jaccard, Paul. "The distribution of the flora in the alpine zone. 1." New phytologist 11.2 (1912): 37-50. <https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1469-8137.1912.tb05611.x>`_
    * `Desgraupes, Bernard. "Clustering indices." University of Paris Ouest-Lab Modal’X 1.1 (2013): 34. <https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11,12,21,22,23

    from permetrics.clustering import ClusteringMetric

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC JACCARD SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 2]
    
    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    js_score = cm.JS()
    print(f"Jaccard Score: {js_score}")

    # ==============================================================================
    # SCENARIO 2: Jaccard vs Rand Score on Highly Dispersed Clusters
    # ==============================================================================
    print("\n--- 2. TRUE NEGATIVE IGNORANCE EXAMPLE ---")

    # When mostly singletons exist, Rand Score stays high due to 'nn', but JS drops
    cm_sparse = ClusteringMetric(y_true=[0, 1, 2, 3, 4], y_pred=[0, 0, 1, 2, 3])

    print(f"Rand Score (Inflated):  {cm_sparse.RaS()}")
    print(f"Jaccard Score (Strict): {cm_sparse.JS()}")
