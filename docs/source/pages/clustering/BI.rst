BI - Beale Index
================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Beale Index (BI)** is an internal clustering evaluation metric based on the statistical F-test :cite:`102307Sparks2346321`. It compares the sum of squared errors of a partition with :math:`K` clusters against a partition with :math:`K-1` clusters (or the overall center) to determine if splitting the data into more clusters significantly reduces the unexplained variance.

Intuitively, BI answers the question: *"Does splitting the data into K clusters provide a statistically significant reduction in variance compared to treating the dataset as a single group?"* A smaller BI score indicates a better clustering partition.

.. math::

    \text{BI} = \frac{\text{WGSS} / (N - K)}{\text{BGSS} / (K - 1)}

Where:

* :math:`N` is the total number of data points (samples).
* :math:`K` is the total number of clusters.
* :math:`\text{WGSS}` is the pooled Within-Group Sum of Squares (unexplained variance), identical to :math:`\text{Tr}(WG)`.
* :math:`\text{BGSS}` is the Between-Group Sum of Squares (explained variance), identical to :math:`\text{Tr}(BG)`.
* The numerator represents the Mean Square Within (:math:`MS_W`).
* The denominator represents the Mean Square Between (:math:`MS_B`).

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The calculation of the Beale Index requires at least two clusters (:math:`K \ge 2`) to evaluate the between-group degrees of freedom (:math:`K - 1`). When evaluated on a single cluster (:math:`K = 1`), the denominator evaluates to zero, making the ratio mathematically undefined.

* **force_finite (bool):** If ``True``, the function catches the undefined division operation when :math:`K = 1` and returns a safe fallback value. Default is ``True``.
* **finite_value (float):** The specific fallback value returned when ``force_finite=True``. Since a smaller score is better for BI, the default fallback is a large penalty value (``1e10``).

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better, indicating optimal cluster compactness relative to separation).
* **Worst possible score:** ``+inf`` (or the defined penalty ``finite_value``).
* **Range:** ``[0.0, +inf)``

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 12,13,22,25

    from permetrics.clustering import ClusteringMetric
    import numpy as np

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC BEALE INDEX EXAMPLE ---")

    X_data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    y_pred_labels = np.array([0, 0, 0, 1, 1, 1])

    cm = ClusteringMetric(X=X_data, y_pred=y_pred_labels)
    bi_score = cm.BI()
    print(f"Beale Index: {bi_score}")

    # ==============================================================================
    # SCENARIO 2: Edge Case with 1 Cluster
    # ==============================================================================
    print("\n--- 2. EDGE CASE (1 CLUSTER) EXAMPLE ---")

    y_pred_single = np.array([0, 0, 0, 0, 0, 0])
    cm_single = ClusteringMetric(X=X_data, y_pred=y_pred_single)

    # Returns the penalty finite_value (1e10) instead of crashing
    bi_safe = cm_single.BI(force_finite=True, finite_value=1e10)
    print(f"BI with 1 cluster (Safe Mode): {bi_safe}")
