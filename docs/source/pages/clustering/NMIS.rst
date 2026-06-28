NMIS - Normalized Mutual Information Score
==========================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Normalized Mutual Information Score (NMIS)** is an external clustering evaluation metric that scales the Mutual Information Score (MIS) to a range of [0, 1]. This normalization addresses the limitation of MIS, where the maximum value depends on the entropy of the partition, making it difficult to interpret or compare across different datasets.

Intuitively, NMIS answers: *"How much information is shared between the ground truth and the prediction, relative to the total entropy of both?"* A value of ``1.0`` indicates perfect agreement between partitions, while ``0.0`` indicates complete independence.

.. math::

    \text{NMIS}(Y, P) = \frac{\text{MIS}(Y, P)}{\text{mean}(\text{H}(Y), \text{H}(P))}

Where:

* :math:`\text{MIS}(Y, P)` is the Mutual Information Score.
* :math:`\text{H}(Y)` and :math:`\text{H}(P)` are the entropy of the true labels and predicted labels, respectively.
* The denominator is the arithmetic mean of the two entropies, which normalizes the score.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The calculation of NMIS becomes undefined if either the ground truth or the predicted labels consist of only a single cluster (i.e., entropy is zero), as this would lead to division by zero.

* **force_finite (bool):** If ``True``, the function catches the undefined operation and returns a safe, finite number instead of raising a ``ValueError``. Default is ``True``.
* **finite_value (float):** The specific fallback value returned when ``force_finite=True`` and the clustering has only 1 cluster. Default is ``0.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates perfect correspondence between the partitions).
* **Worst possible score:** ``0.0`` (Indicates the two partitions share no mutual information).
* **Range:** ``[0.0, 1.0]``
* **References:**

    * `Strehl, Alexander, and Joydeep Ghosh. "Cluster ensembles-a knowledge reuse framework for combining multiple partitions." Journal of machine learning research 3.Dec (2002): 583-617. <https://www.jmlr.org/papers/volume3/strehl02a/strehl02a.pdf>`_
    * `Scikit-Learn Normalized Mutual Information Documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 13,15,24,27

    from permetrics.clustering import ClusteringMetric
    import numpy as np

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC NORMALIZED MUTUAL INFORMATION SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 2, 2]

    # Initialize the metric object
    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    # Calculate the Normalized Mutual Information Score
    nmis_score = cm.NMIS()
    print(f"Normalized Mutual Information Score: {nmis_score}")

    # ==============================================================================
    # SCENARIO 2: Edge Case with 1 Cluster
    # ==============================================================================
    print("\n--- 2. EDGE CASE (1 CLUSTER) EXAMPLE ---")

    # One partition has only 1 cluster, causing undefined entropy
    cm_single = ClusteringMetric(y_true=[0, 0, 0], y_pred=[0, 0, 0])

    # Returns the finite_value (0.0) instead of crashing
    nmis_safe = cm_single.NMIS(force_finite=True, finite_value=0.0)
    print(f"NMIS with 1 cluster (Safe Mode): {nmis_safe}")
