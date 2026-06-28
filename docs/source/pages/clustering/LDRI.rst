LDRI - Log Det Ratio Index
==========================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Log Det Ratio Index (LDRI)** is an internal clustering evaluation metric :cite:`scott1971clustering`. It is a logarithmic variant of the Det-Ratio Index (DRI), scaled by the total number of observations[cite: 397, 401].

Like DRI, it measures the ratio between the determinant of the total scatter matrix (overall data dispersion) and the determinant of the pooled within-cluster scatter matrix (internal cluster dispersion)[cite: 400]. The logarithm smooths out extreme volume differences, and scaling by the number of observations makes the metric comparable across datasets of different sizes.

.. math::

    \text{LDRI} = N \log \left( \frac{|T|}{|W|} \right)

Where:

* :math:`N` is the total number of observations (data points)[cite: 398].
* :math:`|T|` is the determinant of the Total Scatter Matrix :math:`T`[cite: 398, 400].
* :math:`|W|` is the determinant of the pooled Within-Cluster Scatter Matrix :math:`W`

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The calculation of LDRI is susceptible to two mathematical edge cases:

1. **Singular Matrix:** If the within-cluster scatter matrix is singular (e.g., highly collinear features), its determinant :math:`|W|` is 0, causing a division by zero error.
2. **Negative Ratio:** In rare numerical instabilities, if the ratio :math:`|T|/|W| \le 0`, the natural logarithm becomes undefined.

* **force_finite (bool):** If ``True``, the function catches these undefined mathematical operations and returns a safe, finite number instead of raising a ``ValueError``. Default is ``True``.
* **finite_value (float):** The specific fallback value returned when ``force_finite=True`` and the calculation fails. Because a larger score is better for LDRI, the default fallback is a large negative penalty value (``-1e10``).

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``+inf`` (Larger value is better, indicating extremely tight and well-separated clusters).
* **Worst possible score:** ``0.0`` (Since :math:`|T| \ge |W|`, the minimum valid mathematical ratio is 1, and :math:`\log(1) = 0`. However, the penalty score falls back to the defined negative value).
* **Range:** ``[0.0, +inf)`` (or defined penalty bound).

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
    print("--- 1. BASIC LOG DET RATIO INDEX EXAMPLE ---")

    X_data = np.array([[1, 2, 1], [1, 4, 2], [1, 0, 1], [10, 2, 8], [10, 4, 9], [10, 0, 7]])
    y_pred_labels = np.array([0, 0, 0, 1, 1, 1])

    cm = ClusteringMetric(X=X_data, y_pred=y_pred_labels)
    ldri_score = cm.LDRI()
    print(f"Log Det Ratio Index: {ldri_score}")

    # ==============================================================================
    # SCENARIO 2: Edge Case with Singular Matrix (|W| = 0)
    # ==============================================================================
    print("\n--- 2. EDGE CASE (SINGULAR MATRIX) EXAMPLE ---")

    # Highly collinear data where the within-cluster determinant will evaluate to 0
    X_singular = np.array([[1, 1], [2, 2], [3, 3], [10, 10], [11, 11], [12, 12]])
    cm_singular = ClusteringMetric(X=X_singular, y_pred=y_pred_labels)

    # Returns the penalty finite_value (-1e10) instead of crashing
    ldri_safe = cm_singular.LDRI(force_finite=True, finite_value=-1e10)
    print(f"LDRI with Singular Matrix (Safe Mode): {ldri_safe}")
