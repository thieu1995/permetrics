FS - F-Measure Score
=====================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **F-Measure Score (FS)** (widely known in machine learning as the **F1-Score** or **F-beta Score**, and identical to the **Czekanowski-Dice Index**) is an external clustering evaluation metric. It computes the harmonic mean of the pairwise Precision Score (PrS) and Recall Score (ReS).

Intuitively, FS provides a single, unified benchmark to evaluate clustering quality. It answers the question: *"How well does the clustering partition balance both avoiding false-positive co-clusterings (precision) and avoiding fragmented true classes (recall)?"*

.. math::

    \text{FS}_1 = \frac{2 \times \text{PrS} \times \text{ReS}}{\text{PrS} + \text{ReS}} = \frac{2yy}{2yy + yn + ny}

For a generalized weighting parameter :math:`\beta` (where :math:`\beta > 1` favors Recall, and :math:`\beta < 1` favors Precision):

.. math::

    \text{FS}_\beta = \frac{(1 + \beta^2) \times \text{PrS} \times \text{ReS}}{(\beta^2 \times \text{PrS}) + \text{ReS}}

Where across all pairs of distinct data points:

* :math:`yy` (True Positives): Number of pairs placed in the same cluster in both ground truth and prediction.
* :math:`yn` (False Negatives): True intra-class pairs incorrectly split across different predicted clusters.
* :math:`ny` (False Positives): Distinct true classes incorrectly grouped into the same predicted cluster.

-------------------------------------------------------------------------------

Algorithmic Optimizations (Performance Note)
--------------------------------------------

Calculating pairwise harmonized scores via brute-force enumeration scales quadratically at :math:`O(N^2)`. 

This implementation extracts the exact pair totals (:math:`yy`, :math:`yn`, and :math:`ny`) directly from the algebraic dot products of the **Contingency Matrix** marginals. This reduces the computational runtime to **:math:`O(N)`**, allowing instant evaluation on large-scale datasets.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The calculation involves division by the sum of Precision and Recall. If both :math:`\text{PrS}` and :math:`\text{ReS}` evaluate to zero (meaning the model achieved zero true-positive pair groupings, :math:`yy = 0`), the denominator becomes zero, causing an undefined mathematical division.

* **force_finite (bool):** If ``True``, catches the zero-division error and returns a safe fallback value instead of raising a ``ZeroDivisionError``. Default is ``True``.
* **finite_value (float):** The fallback value returned when ``force_finite=True`` and the calculation fails. Since the worst possible valid score is 0.0, the default fallback is ``0.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates identical partitions where precision and recall are both 100%).
* **Worst possible score:** ``0.0`` (The predicted clustering shares zero co-clustered pairs with the ground truth).
* **Permutation Invariance:** The metric is strictly invariant to permutations of cluster labels.
* **Symmetry:** When :math:`\beta = 1.0`, the score is symmetric: :math:`\text{FS}_1(y_{true}, y_{pred}) = \text{FS}_1(y_{pred}, y_{true})`.
* **Mathematical Identity:** :math:`\text{FS}_1 \equiv \text{Czekanowski-Dice Index} \equiv \text{Ochiai Index}`.
* **Range:** ``[0.0, 1.0]``
* **References:** `Desgraupes, Bernard. "Clustering indices." University of Paris Ouest-Lab Modal’X 1.1 (2013): 34. <https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11,12,21

    from permetrics.clustering import ClusteringMetric

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation (Balanced F1-Score)
    # ==============================================================================
    print("--- 1. BASIC F-MEASURE SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 2]
    
    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    fms_score = cm.FS()
    print(f"F-Measure Score (Beta=1.0): {fms_score}")

    # ==============================================================================
    # SCENARIO 2: Favoring Recall via Beta parameter
    # ==============================================================================
    print("\n--- 2. WEIGHTED F-MEASURE EXAMPLE ---")

    # Weigh Recall twice as much as Precision (F2-Score)
    f2_score = cm.FS(beta=2.0)
    print(f"F-Measure Score (Beta=2.0): {f2_score}")
