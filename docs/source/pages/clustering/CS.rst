CS - Completeness Score
=======================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Completeness Score (CS)** is an external clustering evaluation metric based on conditional entropy. A clustering partition satisfies completeness if all data points that are members of a given ground truth class are assigned to the exact same predicted cluster.

Intuitively, CS answers the question: *"Are all samples of class X put into the same cluster?"* A score of ``1.0`` indicates perfectly complete clustering, while ``0.0`` indicates that the cluster assignments fail to group identical classes together.

.. math::

    \text{CS} = 1 - \frac{\text{H}(P | Y)}{\text{H}(P)}

Where:

* :math:`\text{H}(P | Y)` is the conditional entropy of the predicted clusters :math:`P` given the ground truth classes :math:`Y`. It quantifies the remaining uncertainty about which cluster a sample belongs to, given knowledge of its true class.
* :math:`\text{H}(P)` is the entropy of the predicted clusters.

Expressed directly via the Mutual Information Score (:math:`\text{MIS}`):

.. math::

    \text{CS} = \frac{\text{MIS}(Y, P)}{\text{H}(P)}

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The calculation of CS involves division by the entropy of the predicted clusters (:math:`\text{H}(P)`). If the model assigns every single sample into 1 universal cluster (:math:`|P| = 1`), the entropy :math:`\text{H}(P)` evaluates to zero, making the mathematical division undefined.

* **force_finite (bool):** If ``True``, the function catches the zero-division error when :math:`\text{H}(P) = 0` and returns a safe fallback value instead of raising a ``ValueError`` or ``ZeroDivisionError``. Default is ``True``.
* **finite_value (float):** The specific fallback value returned when ``force_finite=True`` and the prediction has only 1 cluster. Since placing all samples into a single cluster trivially guarantees that all members of any true class end up in the same place, the default fallback is ``1.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (All members of any given true class are assigned to the same cluster).
* **Worst possible score:** ``0.0`` (The clustering partition fails to preserve class grouping).
* **Permutation Invariance:** Invariant to permutations of cluster labels.
* **Duality with Homogeneity:** Completeness is the mathematical mirror image of Homogeneity. Specifically:
  
  .. math::

      \text{CS}(y_{true}, y_{pred}) = \text{HS}(y_{pred}, y_{true})

* **Range:** ``[0.0, 1.0]``
* **References:**

    * `Rosenberg, Andrew, and Julia Hirschberg. "V-measure: A conditional entropy-based external cluster evaluation measure." Proceedings of the 2007 joint conference on empirical methods in natural language processing and computational natural language learning (EMNLP-CoNLL). 2007. <https://aclanthology.org/D07-1043.pdf>`_
    * `Scikit-Learn Completeness Score Documentation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11,12,21,22,23

    from permetrics.clustering import ClusteringMetric

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC COMPLETENESS SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    
    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    cs_score = cm.CS()
    print(f"Completeness Score: {cs_score}")

    # ==============================================================================
    # SCENARIO 2: Completeness vs Homogeneity Distinction
    # ==============================================================================
    print("\n--- 2. SINGLE CLUSTER (UNDER-SPLITTING) EXAMPLE ---")

    # Putting all distinct true classes into 1 single cluster gives 100% Completeness
    cm_single = ClusteringMetric(y_true=[0, 1, 2, 3], y_pred=[0, 0, 0, 0])
    print(f"Single Cluster CS: {cm_single.CS()}")
    print(f"Single Cluster HS: {cm_single.HS()}")
