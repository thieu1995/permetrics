EnS - Entropy Score
===================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Entropy Score (EnS)** is an information-theoretic external clustering evaluation metric. Derived from Shannon Entropy, it evaluates the quality of a clustering partition by measuring the degree of class disorder (uncertainty) within each predicted cluster, weighted by the cluster's relative size.

Intuitively, EnS answers the question: "If I randomly pick a data point from predicted cluster :math:`j`, how uncertain am I about its true ground truth class?" A score of ``0.0`` indicates absolute perfection (every cluster contains exclusively samples from a single class), while higher scores indicate chaotic, impure groupings.

.. math::

    \text{EnS} = \sum_{j=1}^{|P|} \frac{|P_j|}{N} \cdot E(P_j)

Where the entropy :math:`E(P_j)` of an individual predicted cluster :math:`j` is defined as:

.. math::

    E(P_j) = - \sum_{i=1}^{|Y|} p_{i, j} \log_2 \left( p_{i, j} \right)

And:

* :math:`N` is the total number of data points.
* :math:`|P_j|` is the number of samples assigned to predicted cluster :math:`j`.
* :math:`p_{i, j} = \frac{C_{i, j}}{|P_j|}` is the empirical probability that a member of predicted cluster :math:`j` belongs to ground truth class :math:`i` (derived directly from the **Contingency Matrix** :math:`C`).

-------------------------------------------------------------------------------

Algorithmic Optimizations (Performance Note)
--------------------------------------------

Naive implementations evaluate cluster entropy by nested iterations over all predicted clusters and true classes, independently slicing and filtering arrays. This incurs an expensive runtime complexity of :math:`O(N \cdot |P| \cdot |Y|)`.

This implementation fully vectorizes the entropy computation. By constructing the global Contingency Matrix, broadcasting the cluster marginal sums, and evaluating the vectorized Shannon entropy formula across non-zero probability elements, it calculates the exact global score in **:math:`O(N)` time complexity**. Furthermore, it safely neutralizes empty clusters and mathematically eliminates undefined :math:`\log_2(0)` operations.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

* **Empty Clusters:** If a predicted cluster contains zero samples (:math:`|P_j| = 0`), its empirical probability distribution is undefined. This implementation explicitly filters out empty partitions prior to matrix division.
* **Empty Datasets:** If the input arrays contain zero elements (:math:`N = 0`), the metric returns ``0.0`` by definition.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Lower is better; indicates zero class uncertainty inside any cluster).
* **Worst possible score:** :math:`\log_2(|Y|)` (Occurs when every single predicted cluster contains a perfectly uniform, equal distribution of all ground truth classes).
* **Permutation Invariance:** Strictly invariant to permutations of cluster labels.
* **Not Symmetric:** In general, :math:`\text{EnS}(y_{true}, y_{pred}) \neq \text{EnS}(y_{pred}, y_{true})`.
* **Duality with Purity:** While Purity evaluates the *mode* (maximum frequency) of a cluster's class distribution, Entropy evaluates the *entire distribution shape*.
* **References:**

    * `Shannon, Claude Elwood. "A mathematical theory of communication." The Bell system technical journal 27.3 (1948): 379-423. <https://doi.org/10.1002/j.1538-7305.1948.tb01338.x>`_
    * `Meilă, Marina. "Comparing clusterings—an information based distance." Journal of multivariate analysis 98.5 (2007): 873-895. <https://doi.org/10.1016/j.jmva.2006.11.013>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11,12,16,17,25,26

    from permetrics.clustering import ClusteringMetric

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation (Perfect vs Impure)
    # ==============================================================================
    print("--- 1. BASIC ENTROPY SCORE EXAMPLE ---")

    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1]

    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    print(f"Perfect Clustering Entropy: {cm.EnS():.4f}")  # Returns 0.0000

    # Introduce chaos (cluster 1 now has mixed classes)
    y_pred_impure = [0, 0, 1, 1, 1, 0]
    cm_impure = ClusteringMetric(y_true=y_true, y_pred=y_pred_impure)
    print(f"Impure Clustering Entropy:  {cm_impure.EnS():.4f}")  # Returns higher value

    # ==============================================================================
    # SCENARIO 2: Maximum Uncertainty Benchmark
    # ==============================================================================
    print("\n--- 2. MAXIMUM UNCERTAINTY EXAMPLE ---")

    # Every predicted cluster is a 50/50 coin flip of class 0 and class 1
    cm_chaos = ClusteringMetric(y_true=[0, 1, 0, 1], y_pred=[0, 0, 1, 1])
    print(f"Maximum Entropy Score: {cm_chaos.EnS():.4f}")  # Returns 2.0
