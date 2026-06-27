GINI - Gini Index
=================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Gini Index** (often referred to as the **Gini Coefficient** or **Accuracy Ratio** in credit scoring) is a measure of a classification model's discriminatory power.

Mathematically, it is a linear transformation of the ROC AUC score. While the ROC AUC represents the absolute area under the curve (ranging from 0.5 to 1.0 for valid models), the Gini Index normalizes this metric so that a random baseline model scores exactly ``0.0``, and a perfect model scores exactly ``1.0``.

.. math::

    \text{Gini} = 2 \times \text{AUC} - 1

-------------------------------------------------------------------------------

Engineering Insight: The Financial Industry Standard
----------------------------------------------------

If you are developing classification models for the banking, insurance, or credit risk sectors, the Gini Index is the mandatory industry standard for model validation.

Business stakeholders often find the baseline AUC score of ``0.5`` (for random guessing) counter-intuitive. By applying the Gini transformation, developers can present a metric where ``0.0`` clearly represents "zero predictive intelligence" and ``1.0`` represents "perfect discrimination," making it significantly easier to communicate model performance to non-technical risk committees.

-------------------------------------------------------------------------------

Averaging Strategies (Via ROC AUC Inheritance)
----------------------------------------------

Because the Gini Index is directly derived from the ROC AUC score, it fully supports the same **One-vs-Rest (OvR)** multiclass decomposition via the `average` keyword argument:

* **None:** Returns a dictionary/array containing the independent Gini score for each class.
* **macro:** Calculates the unweighted arithmetic mean of the OvR Gini scores across all classes.
* **weighted:** Calculates the OvR Gini scores and computes their mean weighted by actual class prevalence.

-------------------------------------------------------------------------------

Benchmark Interpretation Scale
------------------------------

===========  ==================================
Gini Score   Discriminative Power
===========  ==================================
< 0.00       Model is worse than random guessing
0.00         Zero discrimination (Random)
0.01 - 0.40  Weak model
0.41 - 0.60  Good model (Standard for Credit)
0.61 - 0.80  Strong model
> 0.80       Suspiciously high (Check for leakage)
===========  ==================================

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Perfect ranking).
* **Baseline score:** ``0.0`` (Random guessing).
* **Worst possible score:** ``-1.0`` (Systematic inverse ranking).
* **Range:** ``[-1.0, 1.0]``
* **References:** `Scikit-Learn Gini importance (Concepts) <https://scikit-learn.org/stable/modules/tree.html>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 12,13,17,18,32,34-36

    from permetrics.classification import ClassificationMetric

    # ==============================================================================
    # SCENARIO 1: Binary Classification (Passing Probability Scores)
    # y_pred expects continuous probability scores belonging to the Positive Class
    # ==============================================================================
    print("--- 1. BINARY CLASSIFICATION EXAMPLES ---")

    y_true_bin = [0, 0, 1, 1]
    y_score_bin = [0.1, 0.4, 0.35, 0.8]

    cm_bin = ClassificationMetric(y_true_bin, y_score_bin)
    print(f"Binary GINI Score : {cm_bin.GINI()}")

    # Passing a 2D matrix of probabilities
    y_score_2d = [[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]]
    cm_2d = ClassificationMetric(y_true_bin, y_score_2d)
    print(f"Binary GINI (2D)  : {cm_2d.GINI()}")

    # ==============================================================================
    # SCENARIO 2: Multiclass Classification (One-vs-Rest)
    # y_pred expects a 2D array of shape (n_samples, n_classes)
    # ==============================================================================
    print("\n--- 2. MULTICLASS OVR EXAMPLES ---")

    y_true_multi = [0, 1, 2, 0, 1, 2]
    y_score_multi = [
        [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6],
        [0.8, 0.1, 0.1], [0.3, 0.6, 0.1], [0.1, 0.1, 0.8]
    ]

    cm_multi = ClassificationMetric(y_true_multi, y_score_multi)
    # Keyword arguments like 'average' are automatically passed to roc_auc_score
    print(f"average=None (Class dict) : {cm_multi.GINI(average=None)}")
    print(f"average='macro'           : {cm_multi.GINI(average='macro')}")
    print(f"average='weighted'        : {cm_multi.GINI(average='weighted')}")
