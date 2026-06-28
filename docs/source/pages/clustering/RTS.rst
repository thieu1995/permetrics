RTS - Rogers-Tanimoto Score
===========================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Rogers-Tanimoto Score (RTS)** (also known as the **Rogers-Tanimoto Index**) is an external clustering evaluation metric belonging to the pair-counting family. It evaluates the similarity between two partitions by measuring the ratio of concordant pairs to the total pairs, but assigns **double penalty weight** to discordant (mismatched) pairs.

Intuitively, RTS acts as a stricter version of the standard Rand Score. It answers the question: *"If we double the penalty points for every mistake the model makes—whether erroneously grouping separate points or splitting cohesive classes—what is our net pairwise accuracy?"*

.. math::

    \text{RTS} = \frac{yy + nn}{yy + nn + 2(yn + ny)}

Where across all :math:`N_T = \binom{N}{2}` possible pairs of distinct data points:

* :math:`yy` (True Positives): Pairs co-clustered in both partitions.
* :math:`nn` (True Negatives): Pairs separated in both partitions.
* :math:`yn` (False Negatives) and :math:`ny` (False Positives): Discordant pairs (disagreements).

-------------------------------------------------------------------------------

Algorithmic Optimizations (Performance Note)
--------------------------------------------

Standard pairwise comparison requires evaluating all :math:`\binom{N}{2}` sample combinations, resulting in an expensive :math:`O(N^2)` runtime. 

This implementation bypasses explicit pair enumeration. By deriving the exact pair counts directly from the algebraic dot products of the **Contingency Matrix** marginals, it computes the Rogers-Tanimoto index in **:math:`O(N)` time complexity**, guaranteeing optimal memory footprint.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The calculation involves division by :math:`yy + nn + 2(yn + ny)`. Because this denominator is algebraically equal to the total number of sample pairs :math:`N_T` plus the discordant pairs (:math:`N_T + yn + ny`), it can only evaluate to zero if the dataset contains fewer than 2 samples (:math:`N < 2`).

* **force_finite (bool):** If ``True``, catches the zero-division error when :math:`N < 2` and returns a safe fallback value instead of raising a ``ZeroDivisionError``. Default is ``True``.
* **finite_value (float):** The fallback value returned when ``force_finite=True`` and the calculation fails. Since an empty or single-point dataset contains zero meaningful similarity, the default fallback is ``0.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates identical partitions; zero discordant pairs).
* **Worst possible score:** ``0.0`` (Indicates absolute disagreement; zero concordant pairs).
* **Permutation Invariance:** Strictly invariant to permutations of cluster labels.
* **Symmetry:** Strictly symmetric: :math:`\text{RTS}(y_{true}, y_{pred}) = \text{RTS}(y_{pred}, y_{true})`.
* **Relationship with Rand Score (RaS):** Because RTS inflates the denominator by adding an extra set of discordant pairs, it is always strictly bounded by the Rand Score:
  
  .. math::

      \text{RTS} \le \text{RaS}

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
    print("--- 1. BASIC ROGERS-TANIMOTO SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 2]
    
    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    rts_score = cm.RTS()
    print(f"Rogers-Tanimoto Score: {rts_score}")

    # ==============================================================================
    # SCENARIO 2: RTS vs Rand Score Stricter Penalty
    # ==============================================================================
    print("\n--- 2. STRICTER PENALTY COMPARISON ---")

    cm_noisy = ClusteringMetric(y_true=[0, 0, 0, 1, 1], y_pred=[0, 1, 0, 1, 0])

    print(f"Standard Rand Score:     {cm_noisy.RaS():.4f}")
    print(f"Rogers-Tanimoto (Strict): {cm_noisy.RTS():.4f}")
