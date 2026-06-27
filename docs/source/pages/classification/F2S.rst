F2S - F2 Score
==============

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **F2 Score (F2S)** is a specific instance of the generalized :math:`F_\beta` score where :math:`\beta = 2`. It places exactly **twice as much weight on Recall as on Precision**.

.. image:: /_static/images/class_score_1.png
   :align: center
   :alt: F2 Score Weighted Harmonic Mean Illustration

While the F1 score balances false positives and false negatives equally, the F2 score heavily penalizes false negatives. It is the definitive evaluation metric for mission-critical machine learning pipelines where failing to detect a target instance (:math:`FN`) carries far more severe consequences than raising a false alarm (:math:`FP`).

.. math::

    F_2 = (1 + 2^2) \times \frac{\text{Precision} \times \text{Recall}}{(2^2 \times \text{Precision}) + \text{Recall}} = \frac{5 \cdot TP}{5 \cdot TP + 4 \cdot FN + FP}

-------------------------------------------------------------------------------

Averaging Strategies (Multiclass / Multilabel)
----------------------------------------------

When handling more than two classes, the F2 score is computed per class and then aggregated using one of the following `average` parameters:

* **None:** Returns an array/dictionary of independent F2 scores for each individual class.
* **macro:** Calculates the unweighted arithmetic mean of the class F2 scores. It treats rare classes and majority classes with equal importance.
* **micro:** Calculates globally by aggregating the total true positives, false negatives, and false positives across all classes.
* **weighted:** Calculates the F2 score for each class and computes their average weighted by class support, mitigating the distortion of macro-averaging on highly skewed datasets.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates zero False Negatives and perfect Precision).
* **Worst possible score:** ``0.0``
* **Range:** ``[0.0, 1.0]``
* **Engineering Insight (F1 vs. F2 Selection):** Consider an **Automated Defect Detection** vision system on a factory assembly line.
    * If evaluated on **F1 Score**, the model might hesitate to flag ambiguous scratches to keep Precision high.
    * If evaluated on **F2 Score**, the optimization algorithm is forced to flag *everything* remotely suspicious to ensure zero defective products reach customers (:math:`FN \approx 0`), leaving the secondary inspection to human operators.
* **References:** * `Scikit-Learn fbeta_score Documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html>`_,
    * `DeepAI - F-Score Definition <https://deepai.org/machine-learning-glossary-and-terms/f-score>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11,14,18,22,32,34-37,40-43,52,54-57,60-63

    from permetrics.classification import ClassificationMetric

    # ==============================================================================
    # SCENARIO 1: Binary Classification
    # The default 'binary' mode requires a specific positive class (pos_label)
    # ==============================================================================
    print("--- 1. BINARY CLASSIFICATION EXAMPLES ---")

    y_true_bin = [0, 1, 0, 0, 1, 0]
    y_pred_bin = [0, 1, 0, 0, 0, 1]
    cm_bin = ClassificationMetric(y_true_bin, y_pred_bin)

    # 1. Default configuration: average="binary", pos_label=1
    f2_bin_default = cm_bin.F2S()
    print(f"Default (average='binary', pos_label=1): {f2_bin_default}")

    # 2. Change pos_label to 0 (treats 0 as the target positive class)
    f2_bin_pos0 = cm_bin.F2S(average="binary", pos_label=0)
    print(f"Binary with pos_label=0                : {f2_bin_pos0}")

    # 3. When average=None, it returns independent scores for each class found
    f2_bin_none = cm_bin.F2S(average=None)
    print(f"Binary with average=None               : {f2_bin_none}")

    # ==============================================================================
    # SCENARIO 2: Multiclass Classification with Integer Labels
    # ==============================================================================
    print("\n--- 2. MULTICLASS (INTEGER LABELS) EXAMPLES ---")

    y_true_multi_int = [0, 1, 2, 0, 1, 2, 0, 2]
    y_pred_multi_int = [0, 2, 1, 0, 1, 1, 0, 2]
    cm_multi_int = ClassificationMetric(y_true_multi_int, y_pred_multi_int)

    print(f"average=None       : {cm_multi_int.F2S(average=None)}")
    print(f"average='macro'    : {cm_multi_int.F2S(average='macro')}")
    print(f"average='micro'    : {cm_multi_int.F2S(average='micro')}")
    print(f"average='weighted' : {cm_multi_int.F2S(average='weighted')}")

    # Using the `labels` parameter to filter specific classes
    print(f"Filter classes [1, 2] (average=None)    : {cm_multi_int.F2S(labels=[1, 2], average=None)}")
    print(f"Filter classes [1, 2] (average='macro')   : {cm_multi_int.F2S(labels=[1, 2], average='macro')}")
    print(f"Filter classes [1, 2] (average='micro')   : {cm_multi_int.F2S(labels=[1, 2], average='micro')}")
    print(f"Filter classes [1, 2] (average='weighted'): {cm_multi_int.F2S(labels=[1, 2], average='weighted')}")

    # ==============================================================================
    # SCENARIO 3: Multiclass Classification with Categorical/String Labels
    # ==============================================================================
    print("\n--- 3. MULTICLASS (CATEGORICAL/STRING LABELS) EXAMPLES ---")

    y_true_str = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
    y_pred_str = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]
    cm_str = ClassificationMetric(y_true_str, y_pred_str)

    print(f"average=None (Class dict) : {cm_str.F2S(average=None)}")
    print(f"average='macro'           : {cm_str.F2S(average='macro')}")
    print(f"average='micro'           : {cm_str.F2S(average='micro')}")
    print(f"average='weighted'        : {cm_str.F2S(average='weighted')}")

    # Filter string labels: Focus calculation entirely on 'cat' and 'bird'
    print(f"Filter 'cat' & 'bird' (average=None)    : {cm_str.F2S(labels=['cat', 'bird'], average=None)}")
    print(f"Filter 'cat' & 'bird' (average='macro')   : {cm_str.F2S(labels=['cat', 'bird'], average='macro')}")
    print(f"Filter 'cat' & 'bird' (average='micro')   : {cm_str.F2S(labels=['cat', 'bird'], average='micro')}")
    print(f"Filter 'cat' & 'bird' (average='weighted'): {cm_str.F2S(labels=['cat', 'bird'], average='weighted')}")
