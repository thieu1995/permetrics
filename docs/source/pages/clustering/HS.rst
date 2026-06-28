HS - Homogeneity Score
======================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Homogeneity Score (HS)** is an external clustering evaluation metric based on conditional entropy. A clustering partition satisfies homogeneity if all of its clusters contain only data points which are members of a single ground truth class.

Intuitively, HS answers the question: *"Does each predicted cluster contain only samples from a single true class?"* A score of ``1.0`` indicates perfectly homogeneous clustering, while ``0.0`` indicates that the cluster assignments provide no information about the true class labels.

.. math::

    \text{HS} = 1 - \frac{\text{H}(Y | P)}{\text{H}(Y)}

Where:

* :math:`\text{H}(Y | P)` is the conditional entropy of the ground truth classes :math:`Y` given the predicted clusters :math:`P`. It quantifies the remaining uncertainty about the true class of a sample after knowing its assigned cluster.
* :math:`\text{H}(Y)` is the entropy of the ground truth classes.

Expressed directly via the Mutual Information Score (:math:`\text{MIS}`):

.. math::

    \text{HS} = \frac{\text{MIS}(Y, P)}{\text{H}(Y)}

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The calculation of HS involves division by the entropy of the ground truth classes (:math:`\text{H}(Y)`). If the dataset contains only a single ground truth class (:math:`|Y| = 1`), the entropy :math:`\text{H}(Y)` is zero, making the ratio mathematically undefined.

* **force_finite (bool):** If ``True``, the function catches the zero-division error when :math:`\text{H}(Y) = 0` and returns a safe fallback value instead of raising a ``ValueError`` or ``ZeroDivisionError``. Default is ``True``.
* **finite_value (float):** The specific fallback value returned when ``force_finite=True`` and the ground truth has only 1 class. Since a single true class is trivially homogeneous regardless of the predicted clustering, the default fallback is ``1.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Each cluster contains exclusively samples from one ground truth class).
* **Worst possible score:** ``0.0`` (The clustering partition provides zero predictive power regarding the true classes).
* **Permutation Invariance:** The score is completely invariant to permutations of cluster labels.
* **Not Symmetric:** In general, :math:`\text{HS}(y_{true}, y_{pred}) \neq \text{HS}(y_{pred}, y_{true})`. Switching the arguments yields the **Completeness Score (CS)**.
* **Range:** ``[0.0, 1.0]``
* **References:**

    * `Rosenberg, Andrew, and Julia Hirschberg. "V-measure: A conditional entropy-based external cluster evaluation measure." Proceedings of the 2007 joint conference on empirical methods in natural language processing and computational natural language learning (EMNLP-CoNLL). 2007. <https://aclanthology.org/D07-1043.pdf>`_
    * `Scikit-Learn Homogeneity Score Documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11,12,21,22

    from permetrics.clustering import ClusteringMetric

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC HOMOGENEITY SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    
    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    hs_score = cm.HS()
    print(f"Homogeneity Score: {hs_score}")

    # ==============================================================================
    # SCENARIO 2: Homogeneity vs Completeness Distinction
    # ==============================================================================
    print("\n--- 2. OVER-SPLITTING EXAMPLE ---")

    # Splitting one true class into multiple distinct clusters preserves 100% Homogeneity
    cm_oversplit = ClusteringMetric(y_true=[0, 0, 0, 0], y_pred=[0, 1, 2, 3])
    print(f"Oversplit HS: {cm_oversplit.HS()}")
