CDS - Czekanowski-Dice Score
============================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Czekanowski-Dice Score (CDS)** (widely known as the **Sørensen–Dice Coefficient** or **Ochiai Index**, and mathematically identical to the balanced **F1-Score**) is an external clustering evaluation metric. It measures the similarity between two clustering partitions by computing the harmonic mean of the pairwise Precision and Recall.

Intuitively, CDS quantifies the ratio of shared agreements to the total number of co-clustered pairs across both partitions. It answers the question: *"Of all the times either model decided to group two points together, what proportion of those decisions were mutual?"* A score of ``1.0`` indicates identical clustering structures.

.. math::

    \text{CDS} = \frac{2yy}{2yy + yn + ny}

Where across all pairs of distinct data points:

* :math:`yy` (True Positives): Number of pairs assigned to the same cluster in both the ground truth (:math:`y_{true}`) and the prediction (:math:`y_{pred}`).
* :math:`yn` (False Negatives): Pairs co-clustered in :math:`y_{true}`, but split in :math:`y_{pred}`.
* :math:`ny` (False Positives): Pairs co-clustered in :math:`y_{pred}`, but split in :math:`y_{true}`.

Expressed directly via the pairwise Precision (:math:`\text{PrS}`) and Recall (:math:`\text{ReS}`):

.. math::

    \text{CDS} = \frac{2 \times \text{PrS} \times \text{ReS}}{\text{PrS} + \text{ReS}}

-------------------------------------------------------------------------------

Algorithmic Optimizations (Performance Note)
--------------------------------------------

Brute-force iteration over all possible sample pairs scales quadratically at :math:`O(N^2)`, which causes severe bottlenecks on larger datasets.

This implementation derives the exact pair totals (:math:`yy`, :math:`yn`, and :math:`ny`) directly from the algebraic properties of the **Contingency Matrix** marginals. This reduces the runtime complexity to **:math:`O(N)`**, ensuring lightning-fast execution.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The calculation involves division by :math:`2yy + yn + ny`. If both partitions consist entirely of singletons (every cluster has exactly 1 data point), neither model groups any pairs together. The denominator evaluates to zero, triggering an undefined mathematical operation.

* **force_finite (bool):** If ``True``, catches the zero-division error and returns a safe fallback value instead of raising a ``ZeroDivisionError``. Default is ``True``.
* **finite_value (float):** The fallback value returned when ``force_finite=True`` and the calculation fails. Since the worst possible valid score is 0.0, the default fallback is ``0.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates perfect agreement between partitions).
* **Worst possible score:** ``0.0`` (The two partitions share zero co-clustered pairs).
* **Permutation Invariance:** The metric is completely invariant to permutations of cluster labels.
* **Symmetry:** The metric is strictly symmetric: :math:`\text{CDS}(y_{true}, y_{pred}) = \text{CDS}(y_{pred}, y_{true})`.
* **Mathematical Identity:** :math:`\text{CDS} \equiv \text{FmS}_1 \equiv \text{F1-Score}`.
* **Range:** ``[0.0, 1.0]``
* **References:**

    * `Dice, Lee R. "Measures of the amount of ecologic association between species." Ecology 26.3 (1945): 297-302. <https://www.jstor.org/stable/1932409>`_
    * `Desgraupes, Bernard. "Clustering indices." University of Paris Ouest-Lab Modal’X 1.1 (2013): 34. <https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11,12,20,21

    from permetrics.clustering import ClusteringMetric

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC CZEKANOWSKI-DICE SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 2]

    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    cds_score = cm.CDS()
    print(f"Czekanowski-Dice Score: {cds_score}")

    # ==============================================================================
    # SCENARIO 2: Demonstrating Identity with F1-Score
    # ==============================================================================
    print("\n--- 2. IDENTITY CHECK EXAMPLE ---")

    f1_score = cm.FmS(beta=1.0)
    print(f"Are CDS and FmS(beta=1) exactly equal? {np.isclose(cds_score, f1_score)}")
