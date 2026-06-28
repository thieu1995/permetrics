ARS - Adjusted Rand Score
=========================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Adjusted Rand Score (ARS)** is an external clustering evaluation metric. It is the corrected-for-chance version of the Rand Score (RaS).

While the standard Rand Score yields excessively high values (approaching 1.0) when evaluating clusterings with many clusters—even if the partitions are completely random—ARS establishes a baseline expectation. Intuitively, it answers: *"How much better is the agreement between these two partitions than what we would expect from purely random assignment?"*

.. math::

    \text{ARS} = \frac{\text{RaS} - \text{Expected}(\text{RaS})}{\text{Max}(\text{RaS}) - \text{Expected}(\text{RaS})}

Expressed directly via the pair counts derived from the contingency matrix:

.. math::

    \text{ARS} = \frac{2(tp \cdot tn - fn \cdot fp)}{(tp + fn)(fn + tn) + (tp + fp)(fp + tn)}

Where across all pairs of samples:

* :math:`tp` (:math:`yy`): Pairs grouped together in both partitions.
* :math:`tn` (:math:`nn`): Pairs separated in both partitions.
* :math:`fp` (:math:`ny`) and :math:`fn` (:math:`yn`): Discordant pairs.

-------------------------------------------------------------------------------

Algorithmic Optimizations (Overflow Protection)
-----------------------------------------------

When computing the numerator :math:`2(tp \cdot tn - fn \cdot fp)` on large sample sizes, the product of pair counts easily exceeds the 32-bit integer limit, leading to silent numerical overflow and catastrophically wrong scores.

This implementation casts the derived pair counts explicitly to high-capacity Python ``int`` objects prior to the arithmetic evaluation. This guarantees mathematical exactness and stability regardless of dataset scale.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates absolute, identical agreement between partitions).
* **Random baseline:** Scores close to ``0.0`` indicate independent partitions that agree no better than random chance.
* **Worst possible score:** ``-1.0`` (Indicates severe disagreement, lower than random expectation).
* **Permutation Invariance:** Invariant to permutations of cluster labels.
* **Symmetric:** :math:`\text{ARS}(y_{true}, y_{pred}) = \text{ARS}(y_{pred}, y_{true})`.
* **References:** `Scikit-Learn Adjusted Rand Score Documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11,12,21,22

    from permetrics.clustering import ClusteringMetric

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC ADJUSTED RAND SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 2, 1, 2]

    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    ars_score = cm.ARS()
    print(f"Adjusted Rand Score: {ars_score}")

    # ==============================================================================
    # SCENARIO 2: Random Baseline Comparison
    # ==============================================================================
    print("\n--- 2. RANDOM BASELINE EXAMPLE ---")

    # Completely independent random partitions hover near 0.0
    cm_random = ClusteringMetric(y_true=[0, 0, 0, 1, 1, 1], y_pred=[1, 0, 1, 0, 1, 0])
    print(f"Random Partition ARS: {cm_random.ARS()}")
