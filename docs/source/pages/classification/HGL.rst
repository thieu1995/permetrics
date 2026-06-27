HGL - Hinge Loss
================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Hinge Loss (HGL)** is a maximum-margin optimization metric primarily used for training classifiers such as Support Vector Machines (SVMs) :cite:`crammer2001algorithmic`.

Unlike probability-based losses (like Log Loss or Brier Score), Hinge Loss penalizes predictions not only when they are incorrect, but also when they are correct but **not confident enough**. It enforces a strict mathematical boundary: a correct prediction must maintain a safety margin of at least ``1.0`` distance units from the decision hyper-plane to avoid incurring a penalty.

.. math::

    L_{\text{hinge}}(y, \hat{w}) = \max\left(0, \, 1 - y \cdot \hat{w}\right)

-------------------------------------------------------------------------------

Architectural Design: Crammer-Singer Multiclass Generalization
--------------------------------------------------------------

While classical Hinge Loss is restricted to binary labels (:math:`y \in \{-1, +1\}`), ``permetrics`` implements the generalized **Crammer-Singer Multiclass Hinge Loss** formulated over discrete integer targets :math:`y \in \{0, 1, \dots, K-1\}`:

.. math::

    L_{\text{CS}}(y_i, \hat{s}_i) = \max\left(0, \, \max_{k \neq y_i} (\hat{s}_{ik}) - \hat{s}_{i, y_i} + 1\right)

Where:

* :math:`\hat{s}_{i, y_i}` is the raw uncalibrated decision score predicted for the true ground truth class.
* :math:`\max_{k \neq y_i} (\hat{s}_{ik})` is the highest score assigned to any *incorrect* class.

Intuitively, the model incurs zero loss if and only if the score of the true class exceeds the score of the next closest competing class by a margin of at least ``1.0``.

-------------------------------------------------------------------------------

Critical Developer Warning: Raw Logits vs. Probabilities
--------------------------------------------------------

**Do NOT pass standardized probabilities** (e.g., outputs from `.predict_proba()`) to `HGL`. **Hinge Loss assumes an unbounded decision space** :math:`(-\infty, +\infty)`. If normalized probabilities bounded in :math:`[0.0, 1.0]` are supplied, the safety margin condition :math:`(\text{Score}_{\text{true}} - \text{Score}_{\text{false}} \ge 1.0)` becomes mathematically impossible to satisfy reliably. You must supply uncalibrated linear scores or raw decision function outputs (e.g., `.decision_function()` in `scikit-learn` or linear layer outputs before Softmax in PyTorch).

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Lower value is better; true class score safely dominates all incorrect classes by a margin :math:`\ge 1.0`).
* **Worst possible score:** Unbounded (:math:`+\infty`).
* **Range:** :math:`[0.0, +\infty)`
* **Optimizer Note:** HGL is a **Loss** metric. Ensure automated hyperparameter search engines are explicitly configured to *minimize*.
* **References:** `Scikit-Learn hinge_loss <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hinge_loss.html>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 13,14,18,19,34,35

    from permetrics.classification import ClassificationMetric

    # ==============================================================================
    # SCENARIO 1: Binary SVM Decision Scores
    # y_pred expects raw distances from the hyperplane (negative = Class 0, positive = Class 1)
    # ==============================================================================
    print("--- 1. BINARY HINGE LOSS EXAMPLES ---")

    y_true_bin = [0, 1, 1, 0]
    # Confident, correct raw scores
    y_decision_good = [-1.5, 2.2, 1.1, -0.8]

    cm_bin = ClassificationMetric(y_true_bin, y_decision_good)
    print(f"Safe margin HGL : {cm_bin.HGL()}")

    # Borderline correct prediction (Score = 0.2 for Class 1 -> Margin violation penalty!)
    y_decision_unsafe = [-1.5, 0.2, 1.1, -0.8]
    cm_unsafe = ClassificationMetric(y_true_bin, y_decision_unsafe)
    print(f"Unsafe margin HGL: {cm_unsafe.HGL()}")

    # ==============================================================================
    # SCENARIO 2: Multiclass Crammer-Singer Hinge Loss
    # y_pred expects a 2D matrix of uncalibrated logits
    # ==============================================================================
    print("\n--- 2. MULTICLASS HINGE LOSS EXAMPLES ---")

    y_true_multi = [0, 1, 2]
    y_logits_multi = [
        [3.2, 0.5, -1.0],  # Class 0 leads Class 1 by 2.7 (> 1.0 margin -> Loss = 0)
        [0.1, 1.8, 1.2],   # Class 1 leads Class 2 by 0.6 (< 1.0 margin -> Loss = 0.4)
        [-0.5, 2.1, 0.4]   # Incorrect (Class 1 leads Class 2 by 1.7 -> Loss = 2.7)
    ]

    cm_multi = ClassificationMetric(y_true_multi, y_logits_multi)
    print(f"Multiclass CS HGL: {cm_multi.HGL()}")
