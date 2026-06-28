HGS - Hubert Gamma Score
========================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Hubert Gamma Score (HGS)** is an external clustering evaluation metric. It measures the similarity between two clustering partitions by computing the correlation between two binary indicator variables representing whether pairs of samples are co-clustered or separated.

Intuitively, HGS evaluates clustering agreement by treating partition comparison as a quadratic assignment problem. It evaluates the net excess of concordant sample pairs over discordant pairs, answering the question: *"Are two points that are grouped together in the ground truth also consistently placed together by the model?"*

.. math::

    \text{HGS} = \sum_{i < j} X_1(i, j) X_2(i, j) - \mu_{X_1}\mu_{X_2}

Where across all :math:`N_T = \binom{N}{2}` possible pairs of distinct data points:

* :math:`X_1` and :math:`X_2` are binary indicator variables for the ground truth (:math:`y_{true}`) and predicted partition (:math:`y_{pred}`), respectively. Their value is 1 if points :math:`i` and :math:`j` are in the same cluster, and 0 otherwise.
* :math:`\mu_{X_1}` and :math:`\mu_{X_2}` are the expected means of these indicator variables across all pairs.

Expressed directly via the pair counts derived from the contingency matrix:

.. math::

    \text{HGS} = N_T \times yy - (yy + yn)(yy + ny)

Note: In clusterCrit literature, the normalized variant :math:`\hat{\Gamma}` divides this raw value by the standard deviations of the indicator variables, bounding the score between -1 and 1.

-------------------------------------------------------------------------------

Algorithmic Optimizations (Performance Note)
--------------------------------------------

Iterating over all sample combinations to construct the indicator matrices incurs an expensive runtime complexity of :math:`O(N^2)`. 

This implementation bypasses explicit pairwise matrix creation. By extracting the positive pair totals (:math:`yy`, :math:`yn`, and :math:`ny`) directly from the algebraic dot products of the **Contingency Matrix** marginals, it computes the exact Hubert Gamma statistic in **:math:`O(N)` time complexity**. This ensures high-speed benchmarking on massive datasets.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The normalized Hubert Gamma Score involves division by the standard deviations of the indicator variables. If either partition consists exclusively of a single universal cluster or strictly of isolated singletons, the variance of the partition's indicator variable evaluates to zero, triggering an undefined mathematical division.

* **force_finite (bool):** If ``True``, catches the zero-division error and returns a safe fallback value instead of raising a ``ZeroDivisionError``. Default is ``True``.
* **finite_value (float):** The fallback value returned when ``force_finite=True`` and the calculation fails. Since the worst possible normalized score is -1.0, the default fallback is ``0.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** Depends on dataset size for unnormalized HGS; strictly ``1.0`` for normalized :math:`\hat{\Gamma}` (indicating absolute agreement).
* **Worst possible score:** Strictly ``-1.0`` for normalized :math:`\hat{\Gamma}` (indicating severe disagreement).
* **Permutation Invariance:** Completely invariant to permutations of cluster labels.
* **Symmetry:** The metric is strictly symmetric: :math:`\text{HGS}(y_{true}, y_{pred}) = \text{HGS}(y_{pred}, y_{true})`.
* **Range:** ``[-1.0, 1.0]`` (for normalized :math:`\hat{\Gamma}`).
* **References:**

    * `Hubert, Lawrence, and Phipps Arabie. "Comparing partitions." Journal of classification 2.1 (1985): 193-218. <https://link.springer.com/article/10.1007/BF01908075>`_
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
    print("--- 1. BASIC HUBERT GAMMA SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 2]
    
    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    hgs_score = cm.HGS()
    print(f"Hubert Gamma Score: {hgs_score}")

    # ==============================================================================
    # SCENARIO 2: Symmetry Verification
    # ==============================================================================
    print("\n--- 2. SYMMETRY EXAMPLE ---")

    hgs_reverse = cm.HGS(y_true=y_pred, y_pred=y_true)
    print(f"Is HGS exactly symmetric? {np.isclose(hgs_score, hgs_reverse)}")
