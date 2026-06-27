RS - Recall Score
=================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Recall Score (RS)** (also known as **Sensitivity**, **True Positive Rate**, or **Probability of Detection**) is the ratio of correctly predicted positive observations to all actual positive observations.

.. image:: /_static/images/class_score_1.png
   :align: center
   :alt: Recall Score Confusion Matrix Illustration

Intuitively, recall measures the ability of the classifier to find *all* the positive samples. It answers the crucial diagnostic question: *"Of all the samples that were actually positive in the ground truth, what fraction did the model successfully catch?"*

.. math::

    \text{Recall} = \frac{TP}{TP + FN}

Where:

* :math:`TP` (True Positives) is the number of correctly predicted positive instances.
* :math:`FN` (False Negatives) is the number of positive instances that the model missed (incorrectly predicted as negative).

-------------------------------------------------------------------------------

Averaging Strategies (Multiclass / Multilabel)
----------------------------------------------

When handling more than two classes, recall is calculated per class and then aggregated using one of the following `average` parameters:

* **None:** Returns an array/dictionary of independent recall scores for each individual class.
* **macro:** Calculates the unweighted arithmetic mean of the recall scores across all classes. It treats all classes equally, regardless of class frequency.
* **micro:** Calculates globally by aggregating the total true positives and false negatives across all classes: :math:`\frac{\sum TP}{\sum TP + \sum FN}`.
* **weighted:** Calculates recall for each class and computes their average weighted by support (the number of true instances for each class).

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Higher value is better, mathematically indicating zero False Negatives).
* **Worst possible score:** ``0.0``
* **Range:** ``[0.0, 1.0]``
* **Engineering Insight (The Recall Trade-off):** In Machine Learning, there is an inherent trade-off between Precision and Recall.
    * In **Spam Filtering**, you prioritize *Precision* over Recall (it is better to let a spam email hit the inbox than to accidentally send a client's legitimate email to the Spam folder).
    * In **Cancer Detection**, you prioritize *Recall* over Precision (it is infinitely better to flag a healthy patient for a secondary checkup than to miss a cancerous tumor :math:`FN`).
* **References:** * `Scikit-Learn Recall Documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html>`_,
    * `Google Developers - Classification: Precision and Recall <https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall>`_

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
    rs_bin_default = cm_bin.RS()
    print(f"Default (average='binary', pos_label=1): {rs_bin_default}")

    # 2. Change pos_label to 0 (treats 0 as the positive class)
    rs_bin_pos0 = cm_bin.RS(average="binary", pos_label=0)
    print(f"Binary with pos_label=0                : {rs_bin_pos0}")

    # 3. When average=None, it returns independent scores for each class found
    rs_bin_none = cm_bin.RS(average=None)
    print(f"Binary with average=None               : {rs_bin_none}")

    # ==============================================================================
    # SCENARIO 2: Multiclass Classification with Integer Labels
    # ==============================================================================
    print("\n--- 2. MULTICLASS (INTEGER LABELS) EXAMPLES ---")

    y_true_multi_int = [0, 1, 2, 0, 1, 2, 0, 2]
    y_pred_multi_int = [0, 2, 1, 0, 1, 1, 0, 2]
    cm_multi_int = ClassificationMetric(y_true_multi_int, y_pred_multi_int)

    print(f"average=None       : {cm_multi_int.RS(average=None)}")
    print(f"average='macro'    : {cm_multi_int.RS(average='macro')}")
    print(f"average='micro'    : {cm_multi_int.RS(average='micro')}")
    print(f"average='weighted' : {cm_multi_int.RS(average='weighted')}")

    # Using the `labels` parameter to filter specific classes
    print(f"Filter classes [1, 2] (average=None)    : {cm_multi_int.RS(labels=[1, 2], average=None)}")
    print(f"Filter classes [1, 2] (average='macro')   : {cm_multi_int.RS(labels=[1, 2], average='macro')}")
    print(f"Filter classes [1, 2] (average='micro')   : {cm_multi_int.RS(labels=[1, 2], average='micro')}")
    print(f"Filter classes [1, 2] (average='weighted'): {cm_multi_int.RS(labels=[1, 2], average='weighted')}")

    # ==============================================================================
    # SCENARIO 3: Multiclass Classification with Categorical/String Labels
    # ==============================================================================
    print("\n--- 3. MULTICLASS (CATEGORICAL/STRING LABELS) EXAMPLES ---")

    y_true_str = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
    y_pred_str = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]
    cm_str = ClassificationMetric(y_true_str, y_pred_str)

    print(f"average=None (Class dict) : {cm_str.RS(average=None)}")
    print(f"average='macro'           : {cm_str.RS(average='macro')}")
    print(f"average='micro'           : {cm_str.RS(average='micro')}")
    print(f"average='weighted'        : {cm_str.RS(average='weighted')}")

    # Filter string labels: Focus calculation entirely on 'cat' and 'bird'
    print(f"Filter 'cat' & 'bird' (average=None)    : {cm_str.RS(labels=['cat', 'bird'], average=None)}")
    print(f"Filter 'cat' & 'bird' (average='macro')   : {cm_str.RS(labels=['cat', 'bird'], average='macro')}")
    print(f"Filter 'cat' & 'bird' (average='micro')   : {cm_str.RS(labels=['cat', 'bird'], average='micro')}")
    print(f"Filter 'cat' & 'bird' (average='weighted'): {cm_str.RS(labels=['cat', 'bird'], average='weighted')}")
