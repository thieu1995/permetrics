MNS - McNemar Score
===================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **McNemar Score (MNS)** is an external clustering evaluation metric derived from McNemar's non-parametric hypothesis test. In the context of partition comparison, it evaluates clustering quality by measuring the asymmetry between True Negatives (:math:`nn`) and False Positives (:math:`ny`).

Intuitively, MNS answers the question: "When the model decides to separate two points into different clusters, does it do so correctly (:math:`nn`), or is it erroneously breaking apart points that belong to the same true class (:math:`ny`)?" A higher score indicates a favorable balance where correct separations vastly outnumber false-positive co-clusterings.

.. math::

    \text{MNS} = \frac{nn - ny}{\sqrt{nn + ny}}

Where across all pairs of distinct data points:

* :math:`nn` (True Negatives): Number of pairs placed in **different** clusters in both the ground truth (:math:`y_{true}`) and the prediction (:math:`y_{pred}`).
* :math:`ny` (False Positives): Number of pairs placed in **different** classes in :math:`y_{true}`, but incorrectly grouped into the **same** cluster in :math:`y_{pred}`.

-------------------------------------------------------------------------------

Algorithmic Optimizations (Performance Note)
--------------------------------------------

Explicitly constructing pairwise contingency tables across all :math:`\binom{N}{2}` combinations imposes an :math:`O(N^2)` computational bottleneck. 

This implementation extracts the exact pair totals (:math:`nn` and :math:`ny`) directly from the algebraic dot products of the **Contingency Matrix** marginals. This reduces the time complexity to **:math:`O(N)`**, enabling rapid benchmarking on massive datasets.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The calculation of MNS involves division by :math:`\sqrt{nn + ny}`. If the predicted partition consists of a single universal cluster containing all data points (:math:`K = 1`), the model separates zero pairs. Both :math:`nn` and :math:`ny` evaluate to zero, triggering an undefined mathematical division.

* **force_finite (bool):** If ``True``, catches the zero-division error and returns a safe fallback value instead of raising a ``ZeroDivisionError``. Default is ``True``.
* **finite_value (float):** The fallback value returned when ``force_finite=True`` and the calculation fails. Since zero separated pairs yield zero net asymmetry, the default fallback is ``0.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** Depends on total pairs :math:`N_T`; approaches :math:`\sqrt{N_T}` when False Positives (:math:`ny`) approach zero.
* **Worst possible score:** Negative values (when false co-clusterings :math:`ny` severely overwhelm correct separations :math:`nn`).
* **Permutation Invariance:** Completely invariant to permutations of cluster labels.
* **Asymmetric:** Unlike Jaccard or Rand, McNemar is strictly directional: 
  
  .. math::

      \text{MNS}(y_{true}, y_{pred}) \neq \text{MNS}(y_{pred}, y_{true})

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
    print("--- 1. BASIC MCNEMAR SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 2]
    
    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    mns_score = cm.MNS()
    print(f"McNemar Score: {mns_score}")

    # ==============================================================================
    # SCENARIO 2: Demonstrating Asymmetry
    # ==============================================================================
    print("\n--- 2. ASYMMETRY CHECK ---")

    mns_reverse = cm.MNS(y_true=y_pred, y_pred=y_true)
    print(f"Forward MNS: {mns_score:.4f} | Reverse MNS: {mns_reverse:.4f}")
