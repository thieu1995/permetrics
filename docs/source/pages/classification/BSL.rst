BSL - Brier Score Loss
======================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Brier Score Loss (BSL)** :cite:`glenn1950verification` measures the mean squared difference between predicted probability assigned to a set of mutually exclusive outcomes and the actual observed outcome.

Originally developed for weather forecasting validation, the Brier Score is a strictly proper scoring rule. In modern machine learning, it serves as the definitive benchmark for **Probability Calibration** — evaluating not just whether a model correctly classifies an instance, but whether its predicted confidence scores mirror true real-world empirical frequencies.

.. math::

    \text{BSL} = \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} (y_{ik} - \hat{p}_{ik})^2

Where:

* :math:`N` is the total number of evaluated predictions.
* :math:`K` is the total number of discrete classes.
* :math:`\hat{p}_{ik}` is the forecasted probability that sample :math:`i` belongs to class :math:`k`.
* :math:`y_{ik}` is the one-hot encoded ground truth indicator (strictly ``1`` if sample :math:`i` belongs to class :math:`k`, and ``0`` otherwise).

-------------------------------------------------------------------------------

Engineering Insight: Brier Score vs. Log Loss
---------------------------------------------

When auditing probabilistic classifiers, developers are frequently forced to choose between **Log Loss (Cross-Entropy)** and the **Brier Score Loss**.

The fundamental differentiator lies in **outlier penalty behavior**:

* **Log Loss is unbounded** (:math:`[0, +\infty)`): It applies a logarithmic penalty. If the ground truth is Class 1, and a broken model predicts a confidence of ``0.00001``, the loss explodes asymptotically toward infinity. A single catastrophic hallucination can ruin the evaluation metric for an entire benchmark dataset.
* **Brier Score Loss is bounded** (:math:`[0, 1]` or :math:`[0, 2]`): Because it operates on quadratic differences (Mean Squared Error applied to probabilities), its maximum possible penalty for a single prediction is strictly capped. It offers a much more stable, noise-tolerant assessment of overall calibration across noisy production environments.

-------------------------------------------------------------------------------

Architectural Design: Dynamic One-Hot Vectorization
---------------------------------------------------

Unlike standard implementations that demand pre-binarized indicator matrices, ``permetrics`` dynamically infers the target classification space and projects the integer ground truth array into an internal One-Hot matrix at runtime:

1. **Binary Classification:** Accepts either a 1D array of positive-class probabilities (e.g., ``[0.1, 0.8]``) or a explicit 2D complementary matrix (e.g., ``[[0.9, 0.1], [0.2, 0.8]]``).
2. **Multiclass Extension:** Automatically evaluates continuous probability distributions across :math:`K` mutually exclusive classes without requiring external preprocessing pipelines.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Lower value is better; perfect calibration where forecasted probabilities match deterministic reality 100%).
* **Worst possible score:** ``1.0`` (for binary 1D) or ``2.0`` (for unnormalized multiclass one-hot distributions).
* **Range:** ``[0.0, 1.0]`` or ``[0.0, 2.0]``
* **Optimizer Note:** BSL is a **Loss** metric. When configuring automated hyperparameter sweepers (such as `Optuna` or `GridSearchCV`), ensure the direction is explicitly configured to *minimize*.
* **References:** `Scikit-Learn brier_score_loss <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 12,13,17,18,33,34

    from permetrics.classification import ClassificationMetric

    # ==============================================================================
    # SCENARIO 1: Binary Probabilistic Forecasting
    # Evaluating uncalibrated vs well-calibrated confidence scores
    # ==============================================================================
    print("--- 1. BINARY PROBABILITY CALIBRATION ---")

    y_true_bin = [0, 1, 1, 0]
    y_prob_bin = [0.1, 0.9, 0.8, 0.3]  # Highly accurate confidence

    cm_bin = ClassificationMetric(y_true_bin, y_prob_bin)
    print(f"Well-calibrated BSL : {cm_bin.BSL()}")

    # Overconfident, terrible model
    y_prob_bad = [0.9, 0.1, 0.2, 0.8]
    cm_bad = ClassificationMetric(y_true_bin, y_prob_bad)
    print(f"Terrible model BSL  : {cm_bad.BSL()}")

    # ==============================================================================
    # SCENARIO 2: Multiclass Probability Distributions
    # y_pred expects a 2D matrix of shape (n_samples, n_classes)
    # ==============================================================================
    print("\n--- 2. MULTICLASS CALIBRATION EXAMPLES ---")

    y_true_multi = [0, 1, 2]
    y_prob_multi = [
        [0.8, 0.1, 0.1],  # High confidence for Class 0 (Correct)
        [0.1, 0.7, 0.2],  # High confidence for Class 1 (Correct)
        [0.3, 0.3, 0.4]   # Low confidence for Class 2 (Unsure, but correct)
    ]

    cm_multi = ClassificationMetric(y_true_multi, y_prob_multi)
    print(f"Multiclass BSL      : {cm_multi.BSL()}")
