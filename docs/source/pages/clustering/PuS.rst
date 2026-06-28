PuS - Purity Score
==================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Purity Score (PuS)** is a simple, transparent external clustering evaluation metric. It quantifies the extent to which each predicted cluster contains data points belonging primarily to a single ground truth class.

Intuitively, PuS answers the question: *"If we assign each predicted cluster to the ground truth class that appears most frequently within it, what would be our overall classification accuracy across the entire dataset?"*

.. math::

    \text{PuS} = \frac{1}{N} \sum_{j=1}^{|P|} \max_{i} \left( C_{i, j} \right)

Where:

* :math:`N` is the total number of data points.
* :math:`C` is the :math:`|Y| \times |P|` **Contingency Matrix**, where :math:`C_{i, j}` represents the number of samples belonging to true class :math:`i` that were assigned to predicted cluster :math:`j`.
* :math:`\max_{i} (C_{i, j})` extracts the sample count of the majority true class inside cluster :math:`j`.

-------------------------------------------------------------------------------

Algorithmic Optimizations (Performance Note)
--------------------------------------------

Standard textbook implementations compute Purity by iterating over every unique class and cluster label, utilizing boolean array masking. This incurs an inefficient runtime complexity of :math:`O(N \cdot |P| \cdot |Y|)`.

This implementation vectorizes the entire process. By constructing the Contingency Matrix and executing a single column-wise maximum reduction (``np.amax(C, axis=0)``), it evaluates the exact score in **:math:`O(N)` time complexity**. Furthermore, it is strictly safe for arbitrary label formats (strings, UUIDs, or non-consecutive integers).

-------------------------------------------------------------------------------

Theoretical Limitation (The "Singleton Illusion")
-------------------------------------------------

While PuS is easy to interpret, **it cannot be used to trade off the quality of clustering against the number of clusters**.

If a model assigns every single sample into its own individual cluster (:math:`|P| = N`), each cluster contains exactly 1 sample. The majority class count for every cluster trivially equals 1, resulting in an artificial :math:`\text{PuS} = 1.0`. Therefore, Purity should always be evaluated alongside a complexity-penalizing metric like the **Adjusted Rand Score (ARS)** or **Normalized Mutual Information (NMIS)**.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Perfectly pure clusters).
* **Worst possible score:** Bounded below by :math:`\frac{\max_i(|Y_i|)}{N}`. (If the model groups all data into 1 single cluster, the purity simply equals the proportion of the largest true class in the dataset).
* **Permutation Invariance:** Strictly invariant to permutations of cluster labels.
* **Not Symmetric:** In general, :math:`\text{PuS}(y_{true}, y_{pred}) \neq \text{PuS}(y_{pred}, y_{true})`.
* **Range:** ``[0.0, 1.0]``
* **References:** `Manning, Christopher D. Introduction to information retrieval. Syngress Publishing,, 2008. <https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11,12,25,26

    from permetrics.clustering import ClusteringMetric

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC PURITY SCORE EXAMPLE ---")

    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 2, 2]

    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    pus_score = cm.PuS()
    print(f"Purity Score: {pus_score:.4f}")  # Returns 0.8333 (5 out of 6 pure)

    # ==============================================================================
    # SCENARIO 2: Demonstrating the "Singleton Illusion"
    # ==============================================================================
    print("\n--- 2. THE SINGLETON ILLUSION ---")

    # 100 completely random true classes
    y_true_random = [0, 0, 1, 1, 2, 2]
    # Model cheats by putting every point into its own separate cluster
    y_pred_cheating = [10, 11, 12, 13, 14, 15]

    cm_cheat = ClusteringMetric(y_true=y_true_random, y_pred=y_pred_cheating)
    print(f"Cheating Model Purity: {cm_cheat.PuS()}")  # Returns 1.0 (Beware!)
