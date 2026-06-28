DRI - Det-Ratio Index
=====================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Det-Ratio Index (DRI)**, often associated with the Friedman-Rubin criterion :cite:`friedman1967some`, is an internal clustering evaluation metric based on the scatter matrices of the data. It measures the ratio of the determinant of the total scatter matrix to the determinant of the pooled within-cluster scatter matrix.

Intuitively, the determinant of a scatter matrix represents the "volume" or geometric spread of the data. DRI evaluates how much the total data volume expands compared to the sum of the volumes of individual clusters. A higher DRI score indicates that the clusters are tightly packed (small within-cluster volume) and well-separated (large total volume).

.. math::

    \text{DRI} = \frac{|T|}{|W|}

Where:

* :math:`|T|` is the determinant of the Total Scatter Matrix :math:`T`.
* :math:`|W|` is the determinant of the pooled Within-Cluster Scatter Matrix :math:`W` (calculated as :math:`W = \sum_{k=1}^{K} W_k`, where :math:`W_k` is the scatter matrix of the :math:`k`-th cluster).

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The calculation of DRI involves division by the determinant of the within-cluster scatter matrix (:math:`|W|`). If the features within the clusters are perfectly collinear, or if the number of samples is too small, the scatter matrix becomes singular and its determinant :math:`|W|` will be exactly ``0``.

* **force_finite (bool):** If ``True``, the function catches the zero-division error when :math:`|W| = 0` and returns a safe, finite number instead of raising a ``ValueError``. Default is ``True``.
* **finite_value (float):** The specific fallback value returned when ``force_finite=True`` and the determinant is zero. Since a larger score is better for DRI, the default fallback is a large negative penalty value (``-1e10``).

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``+inf`` (Larger value is better, indicating extremely tight and well-separated clusters).
* **Worst possible score:** ``1.0`` (Theoretically, since :math:`T = W + B` and :math:`B` is positive semi-definite, :math:`|T| \ge |W|`. However, the penalty score falls back to the defined negative value).
* **Range:** ``[1.0, +inf)`` (or defined penalty bound).

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 12,13,23,26

    from permetrics.clustering import ClusteringMetric
    import numpy as np

    # ==============================================================================
    # SCENARIO 1: Normal Clustering Evaluation
    # ==============================================================================
    print("--- 1. BASIC DET-RATIO INDEX EXAMPLE ---")

    X_data = np.array([[1, 2, 1], [1, 4, 2], [1, 0, 1], [10, 2, 8], [10, 4, 9], [10, 0, 7]])
    y_pred_labels = np.array([0, 0, 0, 1, 1, 1])

    cm = ClusteringMetric(X=X_data, y_pred=y_pred_labels)
    dri_score = cm.DRI()
    print(f"Det-Ratio Index: {dri_score}")

    # ==============================================================================
    # SCENARIO 2: Edge Case with Singular Matrix (|W| = 0)
    # ==============================================================================
    print("\n--- 2. EDGE CASE (SINGULAR MATRIX) EXAMPLE ---")

    # Highly collinear data where the within-cluster determinant will evaluate to 0
    X_singular = np.array([[1, 1], [2, 2], [3, 3], [10, 10], [11, 11], [12, 12]])
    cm_singular = ClusteringMetric(X=X_singular, y_pred=y_pred_labels)

    # Returns the penalty finite_value (1) instead of crashing
    dri_safe = cm_singular.DRI(force_finite=True, finite_value=1.0)
    print(f"DRI with Singular Matrix (Safe Mode): {dri_safe}")
