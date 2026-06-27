PS - Precision Score
====================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Precision Score (PS)** (also known as **Positive Predictive Value**) is the ratio of correctly predicted positive observations to the total predicted positive observations.

.. image:: /_static/images/class_score_1.png
   :align: center
   :alt: Precision Score Confusion Matrix Illustration

Intuitively, precision represents the ability of the classifier *not* to label a negative sample as positive. It answers the critical question: *"Of all the samples predicted as positive, how many were actually positive?"*

.. math::

    \text{Precision} = \frac{TP}{TP + FP}

Where:

* :math:`TP` (True Positives) is the number of correctly predicted positive instances.
* :math:`FP` (False Positives) is the number of negative instances incorrectly predicted as positive.

-------------------------------------------------------------------------------

Averaging Strategies (Multiclass / Multilabel)
----------------------------------------------

When handling more than two classes, precision is calculated per class and then aggregated using one of the following `average` parameters:

* **None:** Returns an array/dictionary of precision scores for each individual class.
* **macro:** Calculates the unweighted arithmetic mean of the precision scores across all classes. It treats all classes equally, regardless of class frequency.
* **micro:** Calculates globally by aggregating the total true positives and false positives across all classes: :math:`\frac{\sum TP}{\sum TP + \sum FP}`.
* **weighted:** Calculates precision for each class and finds their average weighted by support (the number of true instances for each class). This accounts for class imbalance.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Higher value is better, indicating zero false positives).
* **Worst possible score:** ``0.0``
* **Range:** ``[0.0, 1.0]``
* **References:**
    * `Scikit-Learn Precision Documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html>`_
    * `Neptune.ai - Binary Classification Metrics <https://neptune.ai/blog/evaluation-metrics-binary-classification>`_

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
    ps_bin_default = cm_bin.PS()
    print(f"Default (average='binary', pos_label=1): {ps_bin_default}")

    # 2. Change pos_label to 0 (treats 0 as the positive class)
    ps_bin_pos0 = cm_bin.PS(average="binary", pos_label=0)
    print(f"Binary with pos_label=0                : {ps_bin_pos0}")

    # 3. When average=None, it returns independent scores for each class found
    ps_bin_none = cm_bin.PS(average=None)
    print(f"Binary with average=None               : {ps_bin_none}")

    # ==============================================================================
    # SCENARIO 2: Multiclass Classification with Integer Labels
    # ==============================================================================
    print("\n--- 2. MULTICLASS (INTEGER LABELS) EXAMPLES ---")

    y_true_multi_int = [0, 1, 2, 0, 1, 2, 0, 2]
    y_pred_multi_int = [0, 2, 1, 0, 1, 1, 0, 2]
    cm_multi_int = ClassificationMetric(y_true_multi_int, y_pred_multi_int)

    print(f"average=None       : {cm_multi_int.PS(average=None)}")
    print(f"average='macro'    : {cm_multi_int.PS(average='macro')}")
    print(f"average='micro'    : {cm_multi_int.PS(average='micro')}")
    print(f"average='weighted' : {cm_multi_int.PS(average='weighted')}")

    # Using the `labels` parameter to filter specific classes
    print(f"Filter classes [1, 2] (average=None)    : {cm_multi_int.PS(labels=[1, 2], average=None)}")
    print(f"Filter classes [1, 2] (average='macro')   : {cm_multi_int.PS(labels=[1, 2], average='macro')}")
    print(f"Filter classes [1, 2] (average='micro')   : {cm_multi_int.PS(labels=[1, 2], average='micro')}")
    print(f"Filter classes [1, 2] (average='weighted'): {cm_multi_int.PS(labels=[1, 2], average='weighted')}")

    # ==============================================================================
    # SCENARIO 3: Multiclass Classification with Categorical/String Labels
    # ==============================================================================
    print("\n--- 3. MULTICLASS (CATEGORICAL/STRING LABELS) EXAMPLES ---")

    y_true_str = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
    y_pred_str = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]
    cm_str = ClassificationMetric(y_true_str, y_pred_str)

    print(f"average=None (Class dict) : {cm_str.PS(average=None)}")
    print(f"average='macro'           : {cm_str.PS(average='macro')}")
    print(f"average='micro'           : {cm_str.PS(average='micro')}")
    print(f"average='weighted'        : {cm_str.PS(average='weighted')}")

    # Filter string labels: Focus calculation entirely on 'cat' and 'bird'
    print(f"Filter 'cat' & 'bird' (average=None)    : {cm_str.PS(labels=['cat', 'bird'], average=None)}")
    print(f"Filter 'cat' & 'bird' (average='macro')   : {cm_str.PS(labels=['cat', 'bird'], average='macro')}")
    print(f"Filter 'cat' & 'bird' (average='micro')   : {cm_str.PS(labels=['cat', 'bird'], average='micro')}")
    print(f"Filter 'cat' & 'bird' (average='weighted'): {cm_str.PS(labels=['cat', 'bird'], average='weighted')}")
