RRS - Russell-Rao Score
=======================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Russell-Rao Score (RRS)** (also known as the **Russell-Rao Index**) is an external clustering evaluation metric belonging to the pair-counting family. It quantifies the similarity between two partitions by measuring the exact fraction of sample pairs that are placed in the same cluster across both the ground truth and the prediction relative to the total number of possible pairs.

Intuitively, RRS answers the question: *"Out of all possible pairs of data points in the entire universe of our dataset, what percentage were mutually grouped together by both partitions?"* Unlike the Rand Score or Rogers-Tanimoto, the Russell-Rao Score completely excludes True Negatives (:math:`nn`) from the numerator. This makes it an uninflated, strict measure of positive co-clustering co-occurrence.

.. math::

    \text{RRS} = \frac{yy}{yy + yn + ny + nn} = \frac{yy}{\binom{N}{2}}

Where across all :math:`N_T = \binom{N}{2}` possible pairs of distinct data points:

* :math:`yy` (True Positives): Number of pairs placed in the **same** cluster in both the ground truth (:math:`y_{true}`) and the prediction (:math:`y_{pred}`).
* :math:`yn` (False Negatives), :math:`ny` (False Positives), and :math:`nn` (True Negatives) represent the remaining pairwise combinations.
* The denominator is strictly equal to the total number of sample pairs :math:`N_T`.

-------------------------------------------------------------------------------

Algorithmic Optimizations (Performance Note)
--------------------------------------------

Iterating over all possible sample combinations to count :math:`yy` incurs a quadratic runtime complexity of :math:`O(N^2)`. 

This implementation derives the positive pair total :math:`yy` directly from the algebraic dot products of the **Contingency Matrix** marginals. This reduces the computational complexity to **:math:`O(N)` time**, guaranteeing instantaneous benchmarking even on large-scale datasets.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The calculation involves division by the total number of sample pairs :math:`N_T`. This denominator can only evaluate to zero if the dataset contains fewer than 2 samples (:math:`N < 2`), meaning no pairwise relationships can be formed.

* **force_finite (bool):** If ``True``, catches the zero-division error when :math:`N < 2` and returns a safe fallback value instead of raising a ``ZeroDivisionError``. Default is ``True``.
* **finite_value (float):** The fallback value returned when ``force_finite=True`` and the dataset is too small. Since an empty or single-point dataset contains zero co-clustering probability, the default fallback is ``0.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** Depends on the ground truth structure; strictly equals :math:`1.0` only when the dataset consists of a single universal cluster (:math:`K=1`) and the prediction matches it perfectly.
* **Worst possible score:** ``0.0`` (The two partitions share zero positive co-clustered pairs).
* **Permutation Invariance:** Strictly invariant to permutations of cluster labels.
* **Symmetry:** Strictly symmetric: :math:`\text{RRS}(y_{true}, y_{pred}) = \text{RRS}(y_{pred}, y_{true})`.
* **Relationship with Jaccard Score (JS):** Because RRS uses the global total pairs :math:`N_T` as the denominator instead of just the union of positive pairings, it is always strictly bounded by the Jaccard Score:
  
  .. math::

      \text{RRS} \le \text{JS} \le \text{CDS}

* **Range:** ``[0.0, 1.0]``
* **References:** `Desgraupes, Bernard. "Clustering indices." University of Paris Ouest-Lab Modal’X 1.1 (2013): 34. <https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11,12,22,23

    from permetrics.clustering import ClusteringMetric

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC RUSSELL-RAO SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 2]
    
    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    rrs_score = cm.RRS()
    print(f"Russell-Rao Score: {rrs_score}")

    # ==============================================================================
    # SCENARIO 2: RRS vs Jaccard Score on Sparse Clusters
    # ==============================================================================
    print("\n--- 2. DENOMINATOR DILUTION COMPARISON ---")

    cm_diluted = ClusteringMetric(y_true=[0, 0, 1, 2, 3], y_pred=[0, 0, 1, 2, 3])

    print(f"Jaccard Score (Union base): {cm_diluted.JS():.4f}")
    print(f"Russell-Rao (Global base):  {cm_diluted.RRS():.4f}")
