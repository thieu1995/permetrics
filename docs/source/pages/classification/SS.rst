SS - Specificity Score
======================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Specificity Score (SS)** (also known as **True Negative Rate** or **Selectivity**) is the ratio of correctly predicted negative observations to all actual negative observations.

.. image:: /_static/images/class_score_1.png
   :align: center
   :alt: Specificity Score Confusion Matrix Illustration

Intuitively, specificity measures the ability of the classifier to find *all* the negative samples. It answers the critical evaluation question: *"Of all the samples that were actually negative in the ground truth, what fraction did the model successfully identify as negative?"*

.. math::

    \text{Specificity} = \frac{TN}{TN + FP}

Where:

* :math:`TN` (True Negatives) is the number of correctly predicted negative instances.
* :math:`FP` (False Positives) is the number of actual negative instances incorrectly predicted as positive (Type I error).

-------------------------------------------------------------------------------

Averaging Strategies (Multiclass / Multilabel)
----------------------------------------------

When handling more than two classes, specificity is calculated per class (treating each individual class as a One-vs-Rest problem) and then aggregated using one of the following `average` parameters:

* **None:** Returns an array/dictionary of independent specificity scores for each individual class.
* **macro:** Calculates the unweighted arithmetic mean of the specificity scores across all classes. It treats all classes equally, regardless of class frequency.
* **micro:** Calculates globally by aggregating the total true negatives and false positives across all classes: :math:`\frac{\sum TN}{\sum TN + \sum FP}`.
* **weighted:** Calculates specificity for each class and computes their average weighted by support (the number of true instances for each class).

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Higher value is better, mathematically indicating zero False Positives).
* **Worst possible score:** ``0.0``
* **Range:** ``[0.0, 1.0]``
* **Engineering Insight (Sensitivity vs. Specificity):** Models often face a strict trade-off between Sensitivity (Recall) and Specificity.
    * In **Intrusion Detection Systems (Cybersecurity)**, high *Specificity* ensures that normal employee network traffic isn't constantly flagged as a cyberattack (:math:`FP`), which would overwhelm the security operations center.
    * In **Medical Screening**, a confirmatory test (like a biopsy following a positive blood test) must have a Specificity near ``1.0`` to guarantee that healthy patients are not needlessly subjected to aggressive treatments.
* **References:** * `Wikipedia - Sensitivity and specificity <https://en.wikipedia.org/wiki/Sensitivity_and_specificity>`_,
    * `Neptune.ai - Binary Classification Evaluation Metrics <https://neptune.ai/blog/evaluation-metrics-binary-classification>`_

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
    ss_bin_default = cm_bin.SS()
    print(f"Default (average='binary', pos_label=1): {ss_bin_default}")

    # 2. Change pos_label to 0 (treats 0 as the target positive class)
    ss_bin_pos0 = cm_bin.SS(average="binary", pos_label=0)
    print(f"Binary with pos_label=0                : {ss_bin_pos0}")

    # 3. When average=None, it returns independent scores for each class found
    ss_bin_none = cm_bin.SS(average=None)
    print(f"Binary with average=None               : {ss_bin_none}")

    # ==============================================================================
    # SCENARIO 2: Multiclass Classification with Integer Labels
    # ==============================================================================
    print("\n--- 2. MULTICLASS (INTEGER LABELS) EXAMPLES ---")

    y_true_multi_int = [0, 1, 2, 0, 1, 2, 0, 2]
    y_pred_multi_int = [0, 2, 1, 0, 1, 1, 0, 2]
    cm_multi_int = ClassificationMetric(y_true_multi_int, y_pred_multi_int)

    print(f"average=None       : {cm_multi_int.SS(average=None)}")
    print(f"average='macro'    : {cm_multi_int.SS(average='macro')}")
    print(f"average='micro'    : {cm_multi_int.SS(average='micro')}")
    print(f"average='weighted' : {cm_multi_int.SS(average='weighted')}")

    # Using the `labels` parameter to filter specific classes
    print(f"Filter classes [1, 2] (average=None)    : {cm_multi_int.SS(labels=[1, 2], average=None)}")
    print(f"Filter classes [1, 2] (average='macro')   : {cm_multi_int.SS(labels=[1, 2], average='macro')}")
    print(f"Filter classes [1, 2] (average='micro')   : {cm_multi_int.SS(labels=[1, 2], average='micro')}")
    print(f"Filter classes [1, 2] (average='weighted'): {cm_multi_int.SS(labels=[1, 2], average='weighted')}")

    # ==============================================================================
    # SCENARIO 3: Multiclass Classification with Categorical/String Labels
    # ==============================================================================
    print("\n--- 3. MULTICLASS (CATEGORICAL/STRING LABELS) EXAMPLES ---")

    y_true_str = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
    y_pred_str = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]
    cm_str = ClassificationMetric(y_true_str, y_pred_str)

    print(f"average=None (Class dict) : {cm_str.SS(average=None)}")
    print(f"average='macro'           : {cm_str.SS(average='macro')}")
    print(f"average='micro'           : {cm_str.SS(average='micro')}")
    print(f"average='weighted'        : {cm_str.SS(average='weighted')}")

    # Filter string labels: Focus calculation entirely on 'cat' and 'bird'
    print(f"Filter 'cat' & 'bird' (average=None)    : {cm_str.SS(labels=['cat', 'bird'], average=None)}")
    print(f"Filter 'cat' & 'bird' (average='macro')   : {cm_str.SS(labels=['cat', 'bird'], average='macro')}")
    print(f"Filter 'cat' & 'bird' (average='micro')   : {cm_str.SS(labels=['cat', 'bird'], average='micro')}")
    print(f"Filter 'cat' & 'bird' (average='weighted'): {cm_str.SS(labels=['cat', 'bird'], average='weighted')}")
