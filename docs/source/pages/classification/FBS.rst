FBS - F-Beta Score
==================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **F-Beta Score (FBS)** is the generalized form of the F-score. It serves as a single abstraction layer that weighs Recall :math:`\beta` times as much as Precision, allowing machine learning engineers to explicitly manipulate the evaluation trade-off between False Positives and False Negatives.

.. image:: /_static/images/class_score_1.png
   :align: center
   :alt: F-Beta Score Generalization Illustration


.. math::

    F_\beta = (1 + \beta^2) \times \frac{\text{Precision} \times \text{Recall}}{(\beta^2 \times \text{Precision}) + \text{Recall}} = \frac{(1 + \beta^2) \cdot TP}{(1 + \beta^2) \cdot TP + \beta^2 \cdot FN + FP}

Where :math:`\beta` (beta) is a positive real factor determining the weight of Recall relative to Precision.

-------------------------------------------------------------------------------

Engineering Insight: Tuning the :math:`\beta` Parameter
-------------------------------------------------------

The selection of :math:`\beta` dictates the strict behavioral policy of your optimization target:

* :math:`\beta = 1.0`: Standard harmonic mean (**F1 Score**). False Positives and False Negatives are penalized equally.
* :math:`\beta < 1.0` (e.g., :math:`0.5`): Attaches more weight to **Precision** than Recall. Use this when false alarms (:math:`FP`) are unacceptable (e.g., an automated system recommending expensive, irrevocable stock trades).
* :math:`\beta > 1.0` (e.g., :math:`2.0`): Attaches more weight to **Recall** than Precision. Use this when missed detections (:math:`FN`) are catastrophic (e.g., pedestrian detection in autonomous vehicles).

**Boundary Limits**
	* As :math:`\beta \to 0`, the formula degrades mathematically into pure **Precision**.
	* As :math:`\beta \to +\infty`, the formula degrades mathematically into pure **Recall**.

-------------------------------------------------------------------------------

Averaging Strategies (Multiclass / Multilabel)
----------------------------------------------

When handling more than two classes, the F-beta score is computed per class and then aggregated using one of the following `average` parameters:

* **None:** Returns an array/dictionary of independent F-beta scores for each individual class.
* **macro:** Calculates the unweighted arithmetic mean of the class F-beta scores.
* **micro:** Calculates globally by aggregating the total true positives, false negatives, and false positives across all classes.
* **weighted:** Calculates the F-beta score for each class and computes their average weighted by class support, counteracting the distortion of macro-averaging on highly imbalanced datasets.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0``
* **Worst possible score:** ``0.0``
* **Range:** ``[0.0, 1.0]``
* **References:** `Scikit-Learn fbeta_score Documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html>`_, `Wikipedia - F-score <https://en.wikipedia.org/wiki/F-score>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11,14,16,18,29-32,35-38,47,49-52,55-58

    from permetrics.classification import ClassificationMetric

    # ==============================================================================
    # SCENARIO 1: Binary Classification
    # Testing different Beta weights on the exact same prediction array
    # ==============================================================================
    print("--- 1. BINARY CLASSIFICATION (BETA TUNING) ---")

    y_true_bin = [0, 1, 0, 0, 1, 0]
    y_pred_bin = [0, 1, 0, 0, 0, 1]
    cm_bin = ClassificationMetric(y_true_bin, y_pred_bin)

    # 1. Favor Precision (Beta = 0.5)
    print(f"FBS (beta=0.5) : {cm_bin.FBS(beta=0.5)}")
    # 2. Balanced (Beta = 1.0 -> Identical to F1S)
    print(f"FBS (beta=1.0) : {cm_bin.FBS(beta=1.0)}")
    # 3. Favor Recall (Beta = 2.0 -> Identical to F2S)
    print(f"FBS (beta=2.0) : {cm_bin.FBS(beta=2.0)}")

    # ==============================================================================
    # SCENARIO 2: Multiclass Classification with Integer Labels (beta=1.5)
    # ==============================================================================
    print("\n--- 2. MULTICLASS (INTEGER LABELS) EXAMPLES ---")

    y_true_multi_int = [0, 1, 2, 0, 1, 2, 0, 2]
    y_pred_multi_int = [0, 2, 1, 0, 1, 1, 0, 2]
    cm_multi_int = ClassificationMetric(y_true_multi_int, y_pred_multi_int)

    print(f"average=None       : {cm_multi_int.FBS(beta=1.5, average=None)}")
    print(f"average='macro'    : {cm_multi_int.FBS(beta=1.5, average='macro')}")
    print(f"average='micro'    : {cm_multi_int.FBS(beta=1.5, average='micro')}")
    print(f"average='weighted' : {cm_multi_int.FBS(beta=1.5, average='weighted')}")

    # Filter specific classes with beta=1.5
    print(f"Filter [1, 2] (average=None)    : {cm_multi_int.FBS(beta=1.5, labels=[1, 2], average=None)}")
    print(f"Filter [1, 2] (average='macro')   : {cm_multi_int.FBS(beta=1.5, labels=[1, 2], average='macro')}")
    print(f"Filter [1, 2] (average='micro')   : {cm_multi_int.FBS(beta=1.5, labels=[1, 2], average='micro')}")
    print(f"Filter [1, 2] (average='weighted'): {cm_multi_int.FBS(beta=1.5, labels=[1, 2], average='weighted')}")

    # ==============================================================================
    # SCENARIO 3: Multiclass Classification with String Labels (beta=0.75)
    # ==============================================================================
    print("\n--- 3. MULTICLASS (CATEGORICAL/STRING LABELS) EXAMPLES ---")

    y_true_str = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
    y_pred_str = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]
    cm_str = ClassificationMetric(y_true_str, y_pred_str)

    print(f"average=None       : {cm_str.FBS(beta=0.75, average=None)}")
    print(f"average='macro'    : {cm_str.FBS(beta=0.75, average='macro')}")
    print(f"average='micro'    : {cm_str.FBS(beta=0.75, average='micro')}")
    print(f"average='weighted' : {cm_str.FBS(beta=0.75, average='weighted')}")

    # Filter string labels with beta=0.75
    print(f"Filter 'cat'&'bird' (average=None)    : {cm_str.FBS(beta=0.75, labels=['cat', 'bird'], average=None)}")
    print(f"Filter 'cat'&'bird' (average='macro')   : {cm_str.FBS(beta=0.75, labels=['cat', 'bird'], average='macro')}")
    print(f"Filter 'cat'&'bird' (average='micro')   : {cm_str.FBS(beta=0.75, labels=['cat', 'bird'], average='micro')}")
    print(f"Filter 'cat'&'bird' (average='weighted'): {cm_str.FBS(beta=0.75, labels=['cat', 'bird'], average='weighted')}")
