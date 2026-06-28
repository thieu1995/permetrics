ReS - Recall Score
==================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Recall Score (ReS)** is an external clustering evaluation metric. Complementing Precision, it evaluates the quality of a clustering partition by measuring the proportion of truly co-clustered sample pairs (according to the ground truth) that were successfully placed into the same cluster by the model.

Intuitively, ReS answers the question: *"Of all the pairs of points that are supposed to be grouped together in the real world, what percentage did my model actually manage to find and put into the same cluster?"* A score of ``1.0`` indicates zero false negatives, meaning no true cluster was fragmented across multiple predicted groups.

.. math::

    \text{ReS} = \frac{yy}{yy + yn}

Where across all pairs of distinct data points:

* :math:`yy` (True Positives): Number of pairs placed in the **same** cluster in both the ground truth (:math:`y_{true}`) and the prediction (:math:`y_{pred}`).
* :math:`yn` (False Negatives): Number of pairs placed in the **same** class in :math:`y_{true}`, but incorrectly split into **different** clusters in :math:`y_{pred}`.
* The denominator :math:`yy + yn` represents the total number of true intra-class pairs defined by the ground truth.

Expressed in conditional probability notation (as formulated in clusterCrit):

.. math::

    \text{ReS} = P(gp_2 | gp_1)

Where :math:`gp_1` and :math:`gp_2` represent the events that two points are grouped together in the ground truth and the predicted partition, respectively.

-------------------------------------------------------------------------------

Algorithmic Optimizations (Performance Note)
--------------------------------------------

Standard pairwise recall evaluation requires iterating over all :math:`\binom{N}{2}` combinations, resulting in an expensive :math:`O(N^2)` time complexity. 

This implementation derives the exact pair totals (:math:`yy` and :math:`yn`) directly from the dot products of the **Contingency Matrix** and its marginal sums. This reduces the time complexity to **:math:`O(N)`**, enabling rapid evaluation on large-scale datasets.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The Recall Score involves division by the total number of true intra-class pairs (:math:`yy + yn`). If the ground truth partition consists exclusively of singletons (every true class contains exactly 1 sample), there are zero positive pairs to recall, making the denominator zero and causing an undefined mathematical division.

* **force_finite (bool):** If ``True``, the function catches the zero-division error and returns a safe fallback value instead of raising a ``ZeroDivisionError``. Default is ``True``.
* **finite_value (float):** The fallback value returned when ``force_finite=True`` and the ground truth defines no intra-class pairs. Since having only singletons means the model missed zero positive co-clusterings, the default fallback is ``1.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Higher value is better, indicating that 100% of the true intra-class pairs were successfully recovered).
* **Worst possible score:** ``0.0`` (None of the true intra-class pairs were grouped together by the model).
* **Permutation Invariance:** The score is completely invariant to permutations of cluster labels.
* **Duality with Precision:** Recall is the exact asymmetric counterpart to Precision. Specifically:
  
  .. math::

      \text{ReS}(y_{true}, y_{pred}) = \text{PrS}(y_{pred}, y_{true})

* **Range:** ``[0.0, 1.0]``
* **References:** `Desgraupes, Bernard. "Clustering indices." University of Paris Ouest-Lab Modal’X 1.1 (2013): 34. <https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11,12,21,22,23

    from permetrics.clustering import ClusteringMetric

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC RECALL SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 2]
    
    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    res_score = cm.ReS()
    print(f"Recall Score: {res_score}")

    # ==============================================================================
    # SCENARIO 2: Recall vs Precision Trade-off
    # ==============================================================================
    print("\n--- 2. OVER-GROUPING EXAMPLE ---")

    # Putting all points into 1 single cluster captures 100% of true pairs (Recall = 1.0)
    cm_single = ClusteringMetric(y_true=[0, 0, 1, 1], y_pred=[0, 0, 0, 0])
    print(f"Single Cluster Recall:    {cm_single.ReS()}")  # Returns 1.0
    print(f"Single Cluster Precision: {cm_single.PrS()}")  # Returns 0.333...
