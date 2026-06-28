KS - Kulczynski Score
=====================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Kulczynski Score (KS)** (also known as the **Kulczynski Index**) is an external clustering evaluation metric. It quantifies the similarity between two clustering partitions by calculating the arithmetic mean of the pairwise Precision Score (PrS) and Recall Score (ReS).

Intuitively, KS answers the question: *"When comparing all co-clustered pairs across both partitions, what is the simple average of our clustering precision and clustering recall?"* Unlike the F1-score (harmonic mean) or Fowlkes-Mallows (geometric mean), the arithmetic mean gives equal, linear weight to both precision and recall without heavily penalizing extreme imbalances between the two.

.. math::

    \text{KS} = \frac{1}{2} \left( \frac{yy}{yy + ny} + \frac{yy}{yy + yn} \right)

Where across all pairs of distinct data points:

* :math:`yy` (True Positives): Number of pairs placed in the **same** cluster in both the ground truth (:math:`y_{true}`) and the prediction (:math:`y_{pred}`).
* :math:`yn` (False Negatives): Pairs co-clustered in :math:`y_{true}`, but separated in :math:`y_{pred}`.
* :math:`ny` (False Positives): Pairs co-clustered in :math:`y_{pred}`, but separated in :math:`y_{true}`.

Expressed directly via the pairwise Precision (:math:`\text{PrS}`) and Recall (:math:`\text{ReS}`):

.. math::

    \text{KS} = \frac{\text{PrS} + \text{ReS}}{2}

-------------------------------------------------------------------------------

Algorithmic Optimizations (Performance Note)
--------------------------------------------

Standard pairwise evaluation requires iterating through all :math:`\binom{N}{2}` possible sample combinations, resulting in an expensive :math:`O(N^2)` computational complexity. 

This implementation derives the exact positive pair totals (:math:`yy`, :math:`yn`, and :math:`ny`) directly from the algebraic dot products of the **Contingency Matrix** marginals. This reduces the time complexity to **:math:`O(N)`**, enabling rapid evaluation on large-scale datasets.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The Kulczynski Score involves evaluating the denominators for Precision (:math:`yy + ny`) and Recall (:math:`yy + yn`). If either the ground truth or the predicted partition consists exclusively of isolated singletons (every cluster contains exactly 1 sample), the number of positive intra-cluster pairs evaluates to zero, causing an undefined mathematical division.

* **force_finite (bool):** If ``True``, catches the zero-division error and returns a safe fallback value instead of raising a ``ZeroDivisionError``. Default is ``True``.
* **finite_value (float):** The fallback value returned when ``force_finite=True`` and the calculation fails. Since the worst possible valid score is 0.0, the default fallback is ``0.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates identical partitions where precision and recall are both 100%).
* **Worst possible score:** ``0.0`` (The two partitions share zero co-clustered pairs).
* **Permutation Invariance:** The metric is completely invariant to permutations of cluster labels.
* **Symmetry:** Strictly symmetric: :math:`\text{KS}(y_{true}, y_{pred}) = \text{KS}(y_{pred}, y_{true})`.
* **Relationship with Pythogorean Means:** For any given clustering partition:
  
  .. math::

      \text{FmS}_1 \le \text{FMS} \le \text{KS}

Harmonic Mean :math:`\le` Geometric Mean :math:`\le` Arithmetic Mean.

* **Range:** ``[0.0, 1.0]``
* **References:** `Desgraupes, Bernard. "Clustering indices." University of Paris Ouest-Lab Modal’X 1.1 (2013): 34. <https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 10,11,20-23

    from permetrics.clustering import ClusteringMetric

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC KULCZYNSKI SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 2]
    
    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    ks_score = cm.KS()
    print(f"Kulczynski Score: {ks_score}")

    # ==============================================================================
    # SCENARIO 2: Verifying Arithmetic Mean Identity
    # ==============================================================================
    print("\n--- 2. MEAN IDENTITY CHECK ---")

    prs = cm.PrS()
    res = cm.ReS()
    manual_ks = (prs + res) / 2
    print(f"Are KS and (PrS + ReS)/2 exactly equal? {np.isclose(ks_score, manual_ks)}")
