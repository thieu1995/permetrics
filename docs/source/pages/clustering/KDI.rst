KDI - Ksq-DetW Index
====================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Ksq-DetW Index (KDI)**, closely related to Marriott's criterion, is an internal clustering validation metric. It evaluates the quality of a partition by calculating the determinant of the pooled within-cluster scatter matrix scaled by the square of the number of clusters.

Intuitively, the determinant of the within-cluster scatter matrix (:math:`|W|`) measures the internal "volume" or dispersion of the clusters. By multiplying this volume by :math:`K^2`, the metric introduces a penalty for adding more clusters. It answers the question: *"Are the clusters tightly packed enough to justify the complexity of using K clusters?"* When comparing different values of :math:`K`, a local minimum or a sharp drop in this index often indicates the optimal number of clusters.

.. math::

    \text{KDI} = K^2 \times |W|

Where:

* :math:`K` is the total number of clusters.
* :math:`W` is the pooled within-cluster scatter matrix (the sum of the scatter matrices of all individual clusters).
* :math:`|W|` is the determinant of matrix :math:`W`.

-------------------------------------------------------------------------------

Normalization Strategy
----------------------

The function provides an internal scaling mechanism to prevent the determinant from exploding (overflow) or vanishing (underflow) when dealing with features of vastly different magnitudes:

* **use_normalized (bool):** If ``True`` (default), the pooled scatter matrix :math:`W` undergoes a Min-Max normalization element-wise before its determinant is calculated: 
  :math:`W_{norm} = \frac{W - \min(W)}{\max(W) - \min(W)}`. 
  This ensures the matrix elements are bounded between 0 and 1, providing numerical stability.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** No strictly defined "best" score for a single evaluation. It is typically used comparatively across different values of :math:`K` to find an elbow or minimum.
* **Worst possible score:** No strictly defined worst score.
* **Range:** ``(-inf, +inf)`` (Due to the optional Min-Max normalization, the determinant of the shifted matrix can potentially span across real numbers).
* **References:** `Marriott, F. H. C. (1971). Practical problems in a method of cluster analysis. Biometrics. <https://www.jstor.org/stable/2528615>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 12,14,23

    from permetrics.clustering import ClusteringMetric
    import numpy as np

    # ==============================================================================
    # SCENARIO 1: Normal Evaluation (with Default Normalization)
    # ==============================================================================
    print("--- 1. BASIC KSQ-DETW INDEX EXAMPLE ---")

    X_data = np.array([[1, 2, 1], [1, 4, 2], [1, 0, 1], [10, 2, 8], [10, 4, 9], [10, 0, 7]])
    y_pred_labels = np.array([0, 0, 0, 1, 1, 1])
    
    cm = ClusteringMetric(X=X_data, y_pred=y_pred_labels)
    # Calculates KDI with use_normalized=True by default
    kdi_score_norm = cm.KDI()
    print(f"Ksq-DetW Index (Normalized): {kdi_score_norm}")

    # ==============================================================================
    # SCENARIO 2: Evaluation Without Normalization
    # ==============================================================================
    print("\n--- 2. RAW MATRIX DETERMINANT EXAMPLE ---")

    # If your data is already strictly standardized and you want the raw mathematical determinant
    kdi_score_raw = cm.KDI(use_normalized=False)
    print(f"Ksq-DetW Index (Raw Matrix): {kdi_score_raw}")
