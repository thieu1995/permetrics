FMS - Fowlkes-Mallows Score
===========================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Fowlkes-Mallows Score (FMS)** is an external clustering evaluation metric. It evaluates the similarity between two clustering partitions by calculating the geometric mean of the pairwise Precision and Recall.

Intuitively, FMS answers the question: *"When comparing all pairs of data points, what is the geometric mean of our clustering precision and clustering recall?"* Because it relies on the geometric mean, FMS severely penalizes clusterings where either precision or recall is exceptionally low.

.. math::

    \text{FMS} = \frac{yy}{\sqrt{(yy + yn) \times (yy + ny)}}

Where across all pairs of distinct data points:

* :math:`yy` (True Positives): Number of pairs assigned to the **same** cluster in both :math:`y_{true}` and :math:`y_{pred}`.
* :math:`yn` (False Negatives): Number of pairs assigned to the same cluster in :math:`y_{true}`, but in **different** clusters in :math:`y_{pred}`.
* :math:`ny` (False Positives): Number of pairs assigned to the same cluster in :math:`y_{pred}`, but in **different** clusters in :math:`y_{true}`.

Expressed directly via the pairwise Precision (:math:`\mathcal{P}`) and Recall (:math:`\mathcal{R}`):

.. math::

    \text{FMS} = \sqrt{\mathcal{P} \times \mathcal{R}}

-------------------------------------------------------------------------------

Algorithmic Optimizations (Performance Note)
--------------------------------------------

Iterating over all possible combinations of pairs to evaluate :math:`yy`, :math:`yn`, and :math:`ny` incurs an expensive time and space complexity of :math:`O(N^2)`. 

This implementation bypasses explicit pair enumeration entirely. By leveraging the dot products of the **Contingency Matrix** and its marginal sums, it computes the exact pairwise totals in **:math:`O(N)` time complexity**. This allows benchmarking on large datasets (e.g., :math:`N > 100,000`) without triggering memory spikes.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The Fowlkes-Mallows Score involves division by the square root of the grouped pair products. If either partition contains only isolated singletons (every cluster has exactly 1 data point), the number of intra-cluster pairs evaluates to zero (:math:`yy + yn = 0` or :math:`yy + ny = 0`), causing a zero-division error.

* **force_finite (bool):** If ``True``, the function catches the undefined division operation and returns a safe fallback value instead of raising a ``ZeroDivisionError``. Default is ``True``.
* **finite_value (float):** The fallback value returned when ``force_finite=True`` and the calculation fails. Since the worst possible score is 0.0, the default fallback is ``0.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates identical partitions where precision and recall are both 100%).
* **Worst possible score:** ``0.0`` (Indicates zero agreement; no pair of points shares a cohesive grouping in both partitions).
* **Permutation Invariance:** The metric is completely invariant to permutations of cluster labels.
* **Range:** ``[0.0, 1.0]``
* **References:**

    * `Fowlkes, Edward B., and Colin L. Mallows. "A method for comparing two hierarchical clusterings." Journal of the American statistical association 78.383 (1983): 553-569. <https://www.tandfonline.com/doi/abs/10.1080/01621459.1983.10478008>`_
    * `Scikit-Learn Fowlkes-Mallows Score Documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fowlkes_mallows_score.html>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11,12,21,23

    from permetrics.clustering import ClusteringMetric

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC FOWLKES-MALLOWS SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 2, 1, 2]
    
    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    fms_score = cm.FMS()
    print(f"Fowlkes-Mallows Score: {fms_score}")

    # ==============================================================================
    # SCENARIO 2: Perfect Match vs Completely Independent Partitions
    # ==============================================================================
    print("\n--- 2. EXTREME CASES EXAMPLE ---")

    # Perfect correspondence
    print(f"Perfect Match FMS:  {cm.FMS(y_true=[0, 0, 1], y_pred=[1, 1, 0])}")
    # No shared intra-cluster pairs
    print(f"Disjoint Match FMS: {cm.FMS(y_true=[0, 0, 0], y_pred=[0, 1, 2])}")
