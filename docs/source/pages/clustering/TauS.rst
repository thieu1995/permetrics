TauS - Tau Score
================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Tau Score (TauS)** (adapted from Kendall's Rank Correlation Coefficient :math:`\tau` for partition comparison) is an external clustering evaluation metric designed for complex and mixed-type data. It evaluates clustering quality by measuring the normalized net difference between concordant and discordant sample pairs.

Intuitively, TauS answers the question: *"When treating cluster assignment as a pairwise ranking problem across all data points, what is the exact rank correlation between the ground truth groupings and the model's predicted groupings?"* A score of ``1.0`` indicates absolute concordance (identical partitions).

.. math::

    \text{TauS} = \frac{S_+ - S_-}{\sqrt{N_d \cdot (N_d - t)}}

Where across all :math:`N_d = \binom{N}{2}` possible distinct sample pairs:

* :math:`S_+ = a + d` (Concordant pairs): Pairs placed in the **same** cluster in both partitions (:math:`a`), plus pairs placed in **different** clusters in both partitions (:math:`d`).
* :math:`S_- = b + c` (Discordant pairs): Pairs co-clustered in ground truth but split in prediction (:math:`b`), plus pairs split in ground truth but co-clustered in prediction (:math:`c`).
* :math:`t = a`: The number of mutual co-clustered tie pairs shared by both reference partitions.

-------------------------------------------------------------------------------

Algorithmic Optimizations (Overflow Protection & Speed)
-------------------------------------------------------

Standard textbook formulas evaluate Kendall's Tau by nested pairwise comparisons, scaling at an unfeasible :math:`O(N^2)` time complexity. Furthermore, on large datasets (e.g., :math:`N > 100,000`), the total pair count :math:`N_d` approaches 5 billion, causing standard 32-bit integer arrays to silently overflow into negative values.

This implementation bypasses explicit pair iteration entirely. By executing **combinatorial reductions over the Contingency Matrix** marginals and explicitly casting intermediate totals to high-capacity 64-bit integers, it evaluates the exact Tau correlation in :math:`O(N)` time complexity with guaranteed numerical stability.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The calculation involves division by :math:`\sqrt{N_d \cdot (N_d - t)}`. This denominator evaluates to zero under two specific conditions:

1. **Identical Partitions:** When the predicted clustering perfectly matches the ground truth, all concordant positive pairs are ties (:math:`t = N_d \rightarrow N_d - t = 0`).
2. **Trivial Datasets:** When the input contains fewer than 2 samples (:math:`N < 2 \rightarrow N_d = 0`).

* **force_finite (bool):** If ``True``, catches the zero-division error and returns a safe fallback value instead of raising a ``ZeroDivisionError``. Default is ``True``.
* **finite_value (float):** The fallback value returned when calculation fails. Since identical partitions represent peak theoretical rank correlation, the default fallback is strictly ``1.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates absolute concordance / identical clustering structures).
* **Worst possible score:** ``-1.0`` (Indicates absolute discordance / perfect inverse grouping).
* **Permutation Invariance:** Strictly invariant to permutations of cluster labels.
* **Symmetry:** Strictly symmetric: :math:`\text{TauS}(y_{true}, y_{pred}) = \text{TauS}(y_{pred}, y_{true})`.
* **Range:** ``[-1.0, 1.0]``
* **References:**

    * `Kendall, Maurice G. "A new measure of rank correlation." Biometrika 30.1-2 (1938): 81-93. <https://doi.org/10.2307/2332226>`_
    * `Ahmad, Amir, and Lipika Dey. "A k-mean clustering algorithm for mixed numeric and categorical data." Data & Knowledge Engineering 63.2 (2007): 503-527. <https://doi.org/10.1016/j.datak.2007.03.016>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11,12,20,21

    from permetrics.clustering import ClusteringMetric

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC TAU SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 2]

    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    taus_score = cm.TauS()
    print(f"Tau Score: {taus_score:.4f}")

    # ==============================================================================
    # SCENARIO 2: Verifying Peak Correlation on Identical Inputs
    # ==============================================================================
    print("\n--- 2. IDENTICAL PARTITION CHECK ---")

    cm_perfect = ClusteringMetric(y_true=[0, 1, 2, 3], y_pred=[10, 20, 30, 40])
    print(f"Perfect Match Tau Score: {cm_perfect.TauS()}")
