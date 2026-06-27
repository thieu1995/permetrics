MCC - Matthews Correlation Coefficient
======================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Matthews Correlation Coefficient (MCC)** :cite:`chicco2020advantages` is a statistical rate of Pearson correlation between observed and predicted binary classifications.

.. image:: /_static/images/class_score_1.png
   :align: center
   :alt: Matthews Correlation Coefficient Matrix Illustration


Widely regarded as the single most informative metric for evaluating binary classifiers on heavily imbalanced datasets, MCC takes into account all four values of the confusion matrix (:math:`TP, TN, FP, FN`). It produces a high score only if the classifier achieves good results in both the majority and minority classes.

.. math::

    \text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}

Mathematical Safeguard: If any of the four sums in the denominator equals zero, the denominator becomes zero. By standard statistical convention, MCC is arbitrarily set to ``0.0`` in this scenario, signifying that the model performs no better than random guessing.

-------------------------------------------------------------------------------

Engineering Insight: MCC vs. F1-Score
-------------------------------------

While the F1 score is the industry default, it possesses a fatal structural blind spot: **it completely ignores True Negatives** (:math:`TN`)

Consider a dataset of 100 patients (95 healthy, 5 diseased). A naive classifier predicts *all* 100 patients as healthy.

* **Accuracy:** ``0.95`` *(Deceptively brilliant)*
* **F1-Score:** ``0.00`` *(Successfully catches the bad model)*

Now consider a slightly different flawed model that predicts 10 healthy and 90 diseased. It catches all 5 diseased patients (:math:`TP=5`), but generates 85 false alarms (:math:`FP=85`, :math:`TN=10`).

* **F1-Score:** ``0.105`` *(Creates an illusion of partial success)*
* **MCC:** ``-0.076`` *(Tells the brutal truth: the model is performing worse than a random coin flip).*

Furthermore, MCC is **label-invariant**. If you swap which class is designated as "positive" (e.g., swapping 0 and 1), the F1 score changes drastically, whereas the MCC remains identical.

-------------------------------------------------------------------------------

Averaging Strategies (Multiclass Extension)
-------------------------------------------

In traditional statistical literature, multiclass MCC (Gorodkin's :math:`R_K`) is strictly computed as a single global value. ``permetrics`` extends this concept to support class-wise decomposition:

* **None:** Computes the binary MCC independent for each class (treating it as a One-vs-Rest problem).
* **macro:** Calculates the unweighted arithmetic mean of the One-vs-Rest MCC scores across all classes. This exposes models that achieve a high global Gorodkin correlation simply by ignoring rare classes.
* **micro:** Calculates globally across the aggregate matrix.
* **weighted:** Calculates the class-specific MCC scores and computes their mean weighted by true class support.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``+1.0`` (Perfect prediction).
* **Baseline score:** ``0.0`` (Equivalent to random prediction).
* **Worst possible score:** ``-1.0`` (Total inverse prediction; the classifier systematically predicts the exact opposite of the ground truth).
* **Range:** ``[-1.0, +1.0]``

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
    mcc_bin_default = cm_bin.MCC()
    print(f"Default (average='binary', pos_label=1): {mcc_bin_default}")

    # 2. Change pos_label to 0
    mcc_bin_pos0 = cm_bin.MCC(average="binary", pos_label=0)
    print(f"Binary with pos_label=0                : {mcc_bin_pos0}")

    # 3. When average=None, it returns independent One-vs-Rest MCC per class
    mcc_bin_none = cm_bin.MCC(average=None)
    print(f"Binary with average=None               : {mcc_bin_none}")

    # ==============================================================================
    # SCENARIO 2: Multiclass Classification with Integer Labels
    # ==============================================================================
    print("\n--- 2. MULTICLASS (INTEGER LABELS) EXAMPLES ---")

    y_true_multi_int = [0, 1, 2, 0, 1, 2, 0, 2]
    y_pred_multi_int = [0, 2, 1, 0, 1, 1, 0, 2]
    cm_multi_int = ClassificationMetric(y_true_multi_int, y_pred_multi_int)

    print(f"average=None       : {cm_multi_int.MCC(average=None)}")
    print(f"average='macro'    : {cm_multi_int.MCC(average='macro')}")
    print(f"average='micro'    : {cm_multi_int.MCC(average='micro')}")
    print(f"average='weighted' : {cm_multi_int.MCC(average='weighted')}")

    # Filter specific classes
    print(f"Filter classes [1, 2] (average=None)    : {cm_multi_int.MCC(labels=[1, 2], average=None)}")
    print(f"Filter classes [1, 2] (average='macro')   : {cm_multi_int.MCC(labels=[1, 2], average='macro')}")
    print(f"Filter classes [1, 2] (average='micro')   : {cm_multi_int.MCC(labels=[1, 2], average='micro')}")
    print(f"Filter classes [1, 2] (average='weighted'): {cm_multi_int.MCC(labels=[1, 2], average='weighted')}")

    # ==============================================================================
    # SCENARIO 3: Multiclass Classification with Categorical/String Labels
    # ==============================================================================
    print("\n--- 3. MULTICLASS (CATEGORICAL/STRING LABELS) EXAMPLES ---")

    y_true_str = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
    y_pred_str = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]
    cm_str = ClassificationMetric(y_true_str, y_pred_str)

    print(f"average=None (Class dict) : {cm_str.MCC(average=None)}")
    print(f"average='macro'           : {cm_str.MCC(average='macro')}")
    print(f"average='micro'           : {cm_str.MCC(average='micro')}")
    print(f"average='weighted'        : {cm_str.MCC(average='weighted')}")

    # Filter string labels
    print(f"Filter 'cat' & 'bird' (average=None)    : {cm_str.MCC(labels=['cat', 'bird'], average=None)}")
    print(f"Filter 'cat' & 'bird' (average='macro')   : {cm_str.MCC(labels=['cat', 'bird'], average='macro')}")
    print(f"Filter 'cat' & 'bird' (average='micro')   : {cm_str.MCC(labels=['cat', 'bird'], average='micro')}")
    print(f"Filter 'cat' & 'bird' (average='weighted'): {cm_str.MCC(labels=['cat', 'bird'], average='weighted')}")
