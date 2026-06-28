PhS - Phi Score
===============

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Phi Score (PhS)** (also known as the **Phi Coefficient** or **Mean Square Contingency Coefficient**) is an external clustering evaluation metric. Mathematically identical to the **Pearson Correlation Coefficient** applied to two binary variables, it evaluates clustering quality by measuring the overall linear association between co-clustered and separated sample pairs across both partitions.

Intuitively, PhS answers the question: *"Is there a statistically rigorous correlation between points being grouped together in the ground truth and points being grouped together by the model?"* A score of ``1.0`` indicates absolute positive correlation (identical partitions), while ``0.0`` indicates complete independence.

.. math::

    \text{PhS} = \frac{yy \times nn - yn \times ny}{\sqrt{(yy + yn)(yy + ny)(yn + nn)(ny + nn)}}

Where across all pairs of distinct data points:

* :math:`yy` (True Positives): Pairs co-clustered in both partitions.
* :math:`nn` (True Negatives): Pairs separated in both partitions.
* :math:`yn` (False Negatives) and :math:`ny` (False Positives): Discordant pairs.

-------------------------------------------------------------------------------

Algorithmic Optimizations (Performance Note)
--------------------------------------------

Computing pairwise correlation via brute-force enumeration scales quadratically at :math:`O(N^2)`. 

This implementation derives the exact pair totals directly from the algebraic dot products of the **Contingency Matrix** marginals. This reduces the runtime complexity to **:math:`O(N)`**, allowing high-speed execution on large datasets.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The calculation involves division by the product of the marginal pair totals. If either partition consists exclusively of a single universal cluster (:math:`K = 1`) or entirely of isolated singletons, the variance of co-clusterings evaluates to zero. This makes the denominator zero, causing an undefined correlation.

* **force_finite (bool):** If ``True``, catches the zero-division error and returns a safe fallback value instead of raising a ``ZeroDivisionError``. Default is ``True``.
* **finite_value (float):** The fallback value returned when ``force_finite=True`` and the calculation fails. Since zero variance indicates no meaningful correlation, the default fallback is ``0.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates absolute agreement between partitions).
* **Worst possible score:** ``-1.0`` (Indicates severe inverse correlation).
* **Permutation Invariance:** Strictly invariant to permutations of cluster labels.
* **Symmetry:** Strictly symmetric: :math:`\text{PhS}(y_{true}, y_{pred}) = \text{PhS}(y_{pred}, y_{true})`.
* **Range:** ``[-1.0, 1.0]``
* **References:** `Desgraupes, Bernard. "Clustering indices." University of Paris Ouest-Lab Modal’X 1.1 (2013): 34. <https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11,12,20,21

    from permetrics.clustering import ClusteringMetric

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC PHI SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 2]
    
    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    phs_score = cm.PhS()
    print(f"Phi Score: {phs_score}")

    # ==============================================================================
    # SCENARIO 2: Symmetry Verification
    # ==============================================================================
    print("\n--- 2. SYMMETRY CHECK ---")

    phs_reverse = cm.PhS(y_true=y_pred, y_pred=y_true)
    print(f"Are Forward and Reverse PhS equal? {np.isclose(phs_score, phs_reverse)}")
