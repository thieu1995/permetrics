VMS - V-Measure Score
=====================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **V-Measure Score (VMS)** is an external clustering evaluation metric that calculates the harmonic mean of the Homogeneity Score (HS) and the Completeness Score (CS). It provides a single, balanced score to measure the overall goodness of a clustering partition.

Intuitively, V-Measure acts similarly to the F1-score in classification, but applies to information-theoretic clustering properties. It answers the question: *"How well does the clustering partition maximize both cluster purity (homogeneity) and class assignment coverage (completeness)?"* A score of ``1.0`` indicates a perfect match where all clusters are completely pure and all classes are fully recovered.

.. math::

    \text{VMS} = \frac{(1 + \beta) \times \text{HS} \times \text{CS}}{(\beta \times \text{HS}) + \text{CS}}

By default, :math:`\beta = 1.0`, which assigns equal weight to both components, simplifying the formulation to:

.. math::

    \text{VMS}_1 = \frac{2 \times \text{HS} \times \text{CS}}{\text{HS} + \text{CS}}

Where:

* :math:`\text{HS}` is the Homogeneity Score.
* :math:`\text{CS}` is the Completeness Score.
* :math:`\beta` is the weight parameter. If :math:`\beta > 1`, completeness is weighted more heavily; if :math:`\beta < 1`, homogeneity is favored.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The calculation of VMS involves division by the sum of Homogeneity and Completeness. If both :math:`\text{HS}` and :math:`\text{CS}` are exactly zero (indicating the clustering partition is completely uninformative and random), the denominator becomes zero, causing an undefined mathematical operation.

* **force_finite (bool):** If ``True``, the function catches the zero-division error when :math:`\text{HS} + \text{CS} = 0` and returns a safe fallback value instead of raising a ``ValueError``. Default is ``True``.
* **finite_value (float):** The specific fallback value returned when ``force_finite=True`` and the calculation fails. Since the worst possible valid score is 0.0, the default fallback is ``0.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Perfect matching where clusters are 100% homogeneous and complete).
* **Worst possible score:** ``0.0`` (The clustering partition offers no informational agreement with the ground truth).
* **Permutation Invariance:** Invariant to permutations of cluster labels.
* **Symmetry:** If :math:`\beta = 1.0`, the metric is completely symmetric: :math:`\text{VMS}(y_{true}, y_{pred}) = \text{VMS}(y_{pred}, y_{true})`.
* **Range:** ``[0.0, 1.0]``
* **References:**

    * `Rosenberg, Andrew, and Julia Hirschberg. "V-measure: A conditional entropy-based external cluster evaluation measure." Proceedings of the 2007 joint conference on empirical methods in natural language processing and computational natural language learning (EMNLP-CoNLL). 2007. <https://aclanthology.org/D07-1043.pdf>`_
    * `Scikit-Learn V-Measure Documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11,12,21,22

    from permetrics.clustering import ClusteringMetric

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation (Balanced Beta = 1.0)
    # ==============================================================================
    print("--- 1. BASIC V-MEASURE SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    
    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    vms_score = cm.VMS()
    print(f"V-Measure Score (Beta=1.0): {vms_score}")

    # ==============================================================================
    # SCENARIO 2: Adjusting Beta Weight
    # ==============================================================================
    print("\n--- 2. CUSTOM BETA WEIGHT EXAMPLE ---")

    # Favor Homogeneity by setting beta < 1.0
    vms_custom = cm.VMS(beta=0.5)
    print(f"V-Measure Score (Beta=0.5): {vms_custom}")
