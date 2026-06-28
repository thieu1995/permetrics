SS1S & SS2S - Sokal-Sneath Scores
=================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Sokal-Sneath Scores** represent two distinct external clustering evaluation metrics originating from numerical taxonomy (Sokal & Sneath, 1963). Both belong to the pair-counting family but apply contrasting weighting philosophies to match and mismatch criteria.

-------------------------------------------------------------------------------

1. Sokal-Sneath 1 Score (SS1S)
------------------------------

The **SS1S** (also known as the **Sokal-Sneath Index 1**) quantifies similarity by evaluating co-clustered pairs while completely ignoring True Negatives (:math:`nn`). However, unlike the Jaccard Score, SS1S applies a **double penalty** to all discordant (mismatched) pairs.

Intuitively, SS1S answers: *"If we take the Jaccard index but count every clustering mistake as two strikes instead of one, what is our positive purity?"*

.. math::

    \text{SS1S} = \frac{yy}{yy + 2(yn + ny)}

-------------------------------------------------------------------------------

2. Sokal-Sneath 2 Score (SS2S)
------------------------------

The **SS2S** (also known as the **Sokal-Sneath Index 2** or **Sokal-Sneath Match Coefficient**) evaluates similarity across the entire universe of sample pairs (including True Negatives :math:`nn`). Unlike the standard Rand Score, SS2S assigns **double reward weight** to concordant agreements (or equivalently, halves the penalty of disagreements).

Intuitively, SS2S answers: *"If getting a pair alignment right is worth 1 full point, but getting it wrong only docks 0.5 points, what is our global partition similarity?"*

.. math::

    \text{SS2S} = \frac{yy + nn}{yy + nn + 0.5(yn + ny)} = \frac{2(yy + nn)}{2(yy + nn) + yn + ny}

-------------------------------------------------------------------------------

Algorithmic Optimizations (Performance Note)
--------------------------------------------

Explicitly pairing all :math:`\binom{N}{2}` data points scales quadratically at :math:`O(N^2)`. 

This implementation bypasses brute-force combination generation. By extracting the exact pair distributions (:math:`yy`, :math:`yn`, :math:`ny`, and :math:`nn`) directly from the algebraic dot products of the **Contingency Matrix** marginals, both SS1S and SS2S are computed in **:math:`O(N)` time complexity**.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

* **For SS1S:** The calculation involves division by :math:`yy + 2(yn + ny)`. This becomes undefined if both partitions consist entirely of isolated singletons (zero co-clustered pairs exist).
* **For SS2S:** The calculation involves division by the global total pairs plus matches. It can only become undefined if the dataset contains fewer than 2 samples (:math:`N < 2`).

For both metrics:

* **force_finite (bool):** If ``True``, catches the zero-division error and returns a safe fallback value. Default is ``True``.
* **finite_value (float):** The fallback value returned when calculation fails. Default is ``0.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates identical clustering partitions).
* **Worst possible score:** ``0.0`` (Indicates severe or total disagreement).
* **Permutation Invariance:** Both metrics are strictly invariant to label permutations.
* **Symmetry:** Both metrics are strictly symmetric: :math:`\text{Score}(y_{true}, y_{pred}) = \text{Score}(y_{pred}, y_{true})`.
* **Mathematical Hierarchy of Pair-Counting Metrics:** For any given partition comparison:

  .. math::

      \text{SS1S} \le \text{JS} \le \text{CDS}

  .. math::

      \text{RTS} \le \text{RaS} \le \text{SS2S}

* **Range:** ``[0.0, 1.0]``
* **References:** `Desgraupes, Bernard. "Clustering indices." University of Paris Ouest-Lab Modal’X 1.1 (2013): 34. <https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 12-14,21-23

    from permetrics.clustering import ClusteringMetric
    import numpy as np

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC SOKAL-SNEATH SCORES EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 2]
    
    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    print(f"Sokal-Sneath 1 (Strict Jaccard): {cm.SS1S():.4f}")
    print(f"Sokal-Sneath 2 (Generous Rand):  {cm.SS2S():.4f}")

    # ==============================================================================
    # SCENARIO 2: Demonstrating the Mathematical Hierarchy
    # ==============================================================================
    print("\n--- 2. HIERARCHY VERIFICATION ---")

    cm_test = ClusteringMetric(y_true=[0, 0, 0, 1, 1], y_pred=[0, 1, 0, 1, 0])
    print(f"SS1S ({cm_test.SS1S():.3f}) <= JS ({cm_test.JS():.3f})")
    print(f"RaS ({cm_test.RaS():.3f}) <= SS2S ({cm_test.SS2S():.3f})")
