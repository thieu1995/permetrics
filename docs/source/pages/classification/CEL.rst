CEL - CrossEntropy Loss
=======================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **CrossEntropy Loss (CEL)** :cite:`mao2023cross` (widely known in statistical literature as **Log Loss** or **Multinomial Logistic Loss**) measures the performance of a classification model whose output is a probability distribution between 0 and 1.

.. image:: /_static/images/CLS_CEL.png
   :align: center
   :alt: Cross Entropy Loss Logarithmic Penalty Illustration


Cross-entropy loss increases as the predicted probability diverges from the actual label. It serves as the ubiquitous foundational optimization objective for training deep neural networks via Maximum Likelihood Estimation.

.. math::

    L_{\text{CE}}(Y, \hat{P}) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log(\hat{p}_{ik})

Where:

* :math:`N` is the total number of evaluated samples.
* :math:`K` is the total number of discrete classes.
* :math:`\hat{p}_{ik}` is the forecasted probability that sample :math:`i` belongs to class :math:`k`.
* :math:`y_{ik}` is the true target distribution value.

-------------------------------------------------------------------------------

Architectural Design: Deep Learning Modernization
-------------------------------------------------

**1. Polymorphic Target Ingestion (Hard vs. Soft Targets)** While legacy libraries enforce strictly binarized indicator targets, ``permetrics`` natively ingests both paradigms:

* **Hard Targets:** Accepts standard 1D integer class arrays (e.g., ``[0, 2, 1]``) and internally projects them into deterministic One-Hot matrices.
* **Soft Targets:** Accepts continuous 2D probability distributions (e.g., ``[[0.9, 0.05, 0.05]]``). This unlocks out-of-the-box benchmarking support for modern regularization techniques including **Label Smoothing**, **Mixup**, and **Knowledge Distillation**.

**2. Asymmetric Epsilon Bounding**
To prevent asymptotic hardware crashes caused by the undefined logarithm of zero (:math:`\log(0) = -\infty`), predicted probabilities are strictly lower-bounded by ``self.EPSILON`` (typically :math:`1 \times 10^{-15}`).

Unlike traditional binary implementations, **the upper bound is intentionally left unclipped at ``1.0``**. Because the multiclass formulation contains no complementary :math:`\log(1 - p)` term, preserving exact unity ensures that a mathematically perfect prediction (:math:`\hat{p} = 1.0`) evaluates to a pure zero loss (:math:`\log(1.0) = 0.0`) without accumulating floating-point Epsilon pollution.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Lower value is better; absolute deterministic match between predicted probabilities and target reality).
* **Worst possible score:** Unbounded (:math:`+\infty`).
* **Range:** ``[0.0, +\infty)``
* **Optimizer Note:** CEL is a **Loss** metric. Automated tuning harnesses (e.g., `Ray Tune`, `Optuna`) must be configured to *minimize*.
* **References:** `PyTorch CrossEntropyLoss <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>`_, `Scikit-Learn log_loss <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 12,13,23,24

    from permetrics.classification import ClassificationMetric

    # ==============================================================================
    # SCENARIO 1: Standard Discrete Targets (Hard Labels)
    # y_pred expects a 2D matrix of forecasted class probabilities
    # ==============================================================================
    print("--- 1. STANDARD HARD TARGETS ---")

    y_true_hard = [0, 1, 2]
    y_pred_prob = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]

    cm_hard = ClassificationMetric(y_true_hard, y_pred_prob)
    print(f"Standard CEL : {cm_hard.CEL()}")

    # ==============================================================================
    # SCENARIO 2: Label Smoothing Regularization (Soft Targets)
    # y_true is passed directly as a smoothed probability matrix
    # ==============================================================================
    print("\n--- 2. LABEL SMOOTHED SOFT TARGETS ---")

    y_true_soft = [[0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.10, 0.10, 0.80]]

    cm_soft = ClassificationMetric(y_true_soft, y_pred_prob)
    print(f"Smoothed CEL : {cm_soft.CEL()}")
