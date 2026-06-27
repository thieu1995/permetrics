JSS - Jaccard Similarity Score
==============================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Jaccard Similarity Score (JSS)** (widely celebrated in Computer Vision as **Intersection over Union** or **IoU**) measures the similarity between finite sample sets. In classification benchmarking, it is defined as the size of the intersection divided by the size of the union of the predicted and ground truth label sets.

.. image:: /_static/images/class_score_1.png
   :align: center
   :alt: Jaccard Similarity Intersection over Union Illustration

Intuitively, Jaccard evaluates the pure overlap between what the model predicted and what actually happened, answering: *"Of all unique instances that were either predicted as positive or truly positive, what fraction was mutually agreed upon?"*

.. math::

    J(Y, \hat{Y}) = \frac{|Y \cap \hat{Y}|}{|Y \cup \hat{Y}|} = \frac{TP}{TP + FP + FN}

-------------------------------------------------------------------------------

Engineering Insight: The "IoU" Standard & F1 Twin
-------------------------------------------------

**1. The Computer Vision Benchmark (mIoU)**
While tabular machine learning defaults to F1-Score, Vision models (YOLO, U-Net) mandate Jaccard. When evaluating semantic segmentation masks, calculating One-vs-Rest Jaccard per class and applying `average='macro'` yields the legendary **mean Intersection over Union (mIoU)**.

**2. The Mathematical Twin of F1-Score**
Like the F1-Score, Jaccard structurally **ignores True Negatives** (:math:`TN`). Furthermore, there is a deterministic, monotonic mapping between the two metrics:

.. math::

    J = \frac{F_1}{2 - F_1} \quad \iff \quad F_1 = \frac{2J}{1 + J}

Because of this relationship, ranking a leaderboard of models by F1-Score will produce the exact same model ranking as Jaccard. However, Jaccard possesses a stricter geometric interpretation and penalizes false predictions slightly more aggressively than F1.

-------------------------------------------------------------------------------

Averaging Strategies (Multiclass / Multilabel)
----------------------------------------------

When handling more than two classes, the Jaccard score is computed per individual class and aggregated via the `average` parameter:

* **None:** Returns an array/dictionary of independent Jaccard (IoU) scores for each target class.
* **macro:** Calculates the unweighted arithmetic mean of the class IoU scores (**mIoU**). It treats rare classes and frequent classes with equal weight.
* **micro:** Calculates globally by aggregating total true positives, false positives, and false negatives: :math:`\frac{\sum TP}{\sum TP + \sum FP + \sum FN}`.
* **weighted:** Calculates the class-wise Jaccard scores and computes their average weighted by true class support.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Perfect overlap; predicted set equals ground truth set).
* **Worst possible score:** ``0.0`` (Zero overlap; predicted set and ground truth set are disjoint).
* **Range:** ``[0.0, 1.0]``
* **References:** `Scikit-Learn <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html>`_, `Wikipedia - Jaccard index <https://en.wikipedia.org/wiki/Jaccard_index>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11,14,18,22,32,34-37,40-43,52,54-57

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
    jss_bin_default = cm_bin.JSS()
    print(f"Default (average='binary', pos_label=1): {jss_bin_default}")

    # 2. Change pos_label to 0
    jss_bin_pos0 = cm_bin.JSS(average="binary", pos_label=0)
    print(f"Binary with pos_label=0                : {jss_bin_pos0}")

    # 3. Independent IoU per class
    jss_bin_none = cm_bin.JSS(average=None)
    print(f"Binary with average=None               : {jss_bin_none}")

    # ==============================================================================
    # SCENARIO 2: Multiclass Classification with Integer Labels
    # ==============================================================================
    print("\n--- 2. MULTICLASS (INTEGER LABELS) EXAMPLES ---")

    y_true_multi_int = [0, 1, 2, 0, 1, 2, 0, 2]
    y_pred_multi_int = [0, 2, 1, 0, 1, 1, 0, 2]
    cm_multi_int = ClassificationMetric(y_true_multi_int, y_pred_multi_int)

    print(f"average=None       : {cm_multi_int.JSS(average=None)}")
    print(f"average='macro'    : {cm_multi_int.JSS(average='macro')}")
    print(f"average='micro'    : {cm_multi_int.JSS(average='micro')}")
    print(f"average='weighted' : {cm_multi_int.JSS(average='weighted')}")

    # Filter specific classes
    print(f"Filter classes [1, 2] (average=None)    : {cm_multi_int.JSS(labels=[1, 2], average=None)}")
    print(f"Filter classes [1, 2] (average='macro')   : {cm_multi_int.JSS(labels=[1, 2], average='macro')}")
    print(f"Filter classes [1, 2] (average='micro')   : {cm_multi_int.JSS(labels=[1, 2], average='micro')}")
    print(f"Filter classes [1, 2] (average='weighted'): {cm_multi_int.JSS(labels=[1, 2], average='weighted')}")

    # ==============================================================================
    # SCENARIO 3: Multiclass Classification with Categorical/String Labels
    # ==============================================================================
    print("\n--- 3. MULTICLASS (CATEGORICAL/STRING LABELS) EXAMPLES ---")

    y_true_str = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
    y_pred_str = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]
    cm_str = ClassificationMetric(y_true_str, y_pred_str)

    print(f"average=None (Class dict) : {cm_str.JSS(average=None)}")
    print(f"average='macro'           : {cm_str.JSS(average='macro')}")
    print(f"average='micro'           : {cm_str.JSS(average='micro')}")
    print(f"average='weighted'        : {cm_str.JSS(average='weighted')}")
