GINI - Regression Gini
======================

.. toctree::
   :maxdepth: 3
   :caption: GINI - Gini coefficient

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


In regression analysis, the term "Gini" refers to two fundamentally different statistical paradigms depending on whether
one evaluates the **ranking capability** of the predictions or the **dispersion of the prediction errors**.

To prevent statistical misinterpretation, `permetrics` explicitly separates these into two distinct metrics:

.. contents:: Table of Contents
   :local:
   :depth: 2

-------------------------------------------------------------------------------

1. Normalized Gini Coefficient (Ranking Power)
----------------------------------------------

The **Normalized Gini Coefficient** :cite:`frees2011summarizing` measures the *actuarial ranking capability* of a regression model.
Inherited from economics (the Lorenz curve) and heavily utilized in insurance pricing, credit scoring, and algorithmic
trading, it quantifies how well the predicted values :math:`y_{\text{pred}}` can rank the actual continuous targets :math:`y_{\text{true}}`.

.. math::

    G_{\text{norm}} = \frac{\text{Gini}(y_{\text{true}}, y_{\text{pred}})}{\text{Gini}(y_{\text{true}}, y_{\text{true}})}

where the numerator is the raw covariance Gini of the model, and the denominator is the raw Gini of an *optimal model* (the ground truth sorted by itself).

Properties
~~~~~~~~~~
* **Best possible score:** ``1.0`` (Perfect ranking: the model sorts targets in the exact correct order).
* **Worst possible score:** ``0.0`` (Random ranking) or ``-1.0`` (Perfectly inverted ranking).
* **Range:** ``[-1, 1]``
* **Function call:** ``evaluator.normalized_gini_coefficient()``


-------------------------------------------------------------------------------

2. Residual Gini Index (Error Dispersion)
-----------------------------------------

The **Residual Gini Index** :cite:`yitzhaki2012gini` applies the classic economic Gini index of inequality to
the **absolute regression residuals** :math:`E = \lvert y_{\text{true}} - y_{\text{pred}} \rvert`.

Instead of measuring ranking, it answers an econometric question: *"Is the model's total error distributed equally
across all samples, or is 90% of the total error caused by 3 extreme outliers?"*

.. math::

    G_{\text{residual}} = \frac{2 \sum_{i=1}^{n} i \cdot e_{(i)}}{n \sum_{i=1}^{n} e_i} - \frac{n+1}{n}

where :math:`e_{(i)}` represents the absolute errors sorted in **non-decreasing order** (:math:`e_{(1)} \le e_{(2)} \le \dots \le e_{(n)}`).

Properties
~~~~~~~~~~
* **Best possible score:** ``0.0`` (Complete equality: every single sample in the dataset experiences the exact same magnitude of error).
* **Worst possible score:** approaching ``1.0`` (Extreme disparity: the model predicts perfectly for almost all samples, but fails catastrophically on a tiny fraction).
* **Range:** ``[0, 1]``
* **Function call:** ``evaluator.residual_gini_index()``

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python

    import numpy as np
    from permetrics.regression import RegressionMetric

    y_true = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([12, 18, 33, 39, 55])

    evaluator = RegressionMetric(y_true, y_pred)

    # 1. Evaluate Ranking capability
    gini_rank = evaluator.normalized_gini_coefficient()
    print(f"Ranking Gini: {gini_rank:.4f}")  # Expected: ~1.0 (Very good ranker)

    # 2. Evaluate Error sparsity
    gini_err = evaluator.residual_gini_index()
    print(f"Residual Gini: {gini_err:.4f}")  # Expected: closer to 0.0
