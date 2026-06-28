GPS - G-Plus Score
==================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **G-Plus Score (GPS)** (originally **Rohlf's :math:`G^+` index**, derived from Goodman & Kruskal statistics) is an external clustering evaluation metric. It evaluates clustering loss by measuring the exact proportion of strictly discordant sample pairs relative to the total number of distinct pairs.

Intuitively, GPS flips the validation paradigm from *measuring structural agreement* to *measuring structural contradiction*. It answers the question: *"Out of all possible pairs of data points, what exact fraction did my model get completely wrong?"*

.. math::

    \text{GPS} = \frac{S_-}{N_d}

Where across all :math:`N_d = \binom{N}{2}` possible pairs of distinct data points:

* :math:`S_- = b + c` (Discordant pairs): Pairs grouped together in the ground truth but split across different clusters in the prediction (:math:`b`), plus pairs separated in the ground truth but incorrectly co-clustered by the model (:math:`c`).

-------------------------------------------------------------------------------

The Mathematical Identity (Relationship with Rand Score)
--------------------------------------------------------

In external validation, the total number of pairs is the strict sum of concordant agreements (:math:`S_+`) and discordant disagreements (:math:`S_-`). Therefore: :math:`S_+ + S_- = N_d \rightarrow S_- = N_d - S_+`.

Recalling that the standard **Rand Score (RaS)** is defined as :math:`\frac{S_+}{N_d}`, the metric collapses into an exact linear complement:

.. math::

    \text{GPS} = \frac{N_d - S_+}{N_d} = 1 - \frac{S_+}{N_d} = 1 - \text{RaS}

**Takeaway:** Unlike almost every other validation index where ``1.0`` represents perfection, **GPS is a pure loss metric**. A score of ``0.0`` indicates zero clustering contradictions (absolute structural perfection), while ``1.0`` indicates total disagreement.

-------------------------------------------------------------------------------

Algorithmic Optimizations (Performance Note)
--------------------------------------------

Iterating through all :math:`\binom{N}{2}` sample combinations to tally discordant pairs scales quadratically at :math:`O(N^2)`.

Leveraging the algebraic identity :math:`\text{GPS} \equiv 1 - \text{RaS}`, this implementation completely bypasses pairwise combinatorial enumeration. It extracts the exact loss metric directly from the :math:`O(N)` contingency table marginals, ensuring maximum execution speed.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The calculation involves division by the total pair count :math:`N_d`. This denominator can only evaluate to zero if the input dataset contains fewer than 2 samples (:math:`N < 2`).

* **force_finite (bool):** If ``True``, catches the zero-division error when :math:`N < 2` and returns a safe fallback value instead of raising a ``ZeroDivisionError``. Default is ``True``.
* **finite_value (float):** The fallback value returned when calculation fails. Since a 1-sample dataset contains zero possible contradiction errors, the default fallback is strictly ``0.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Indicates zero discordant pairs / identical partitions).
* **Random baseline:** ``0.5`` (Independent, random partitions yield a 50% contradiction rate).
* **Worst possible score:** ``1.0`` (Indicates 100% structural disagreement).
* **Permutation Invariance:** Strictly invariant to permutations of cluster labels.
* **Symmetry:** Strictly symmetric: :math:`\text{GPS}(y_{true}, y_{pred}) = \text{GPS}(y_{pred}, y_{true})`.
* **Mathematical Identity:** :math:`\text{GPS} \equiv 1 - \text{RaS} \equiv \text{Rohlf } G^+`.
* **Range:** ``[0.0, 1.0]``
* **References:** `Goodman, Leo A., and William H. Kruskal. "Measures of association for cross classifications." Journal of the American statistical association 49.268 (1954): 732-764. <https://doi.org/10.2307/2281536>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11,12,19,21-23

    from permetrics.clustering import ClusteringMetric

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC G-PLUS SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 2, 2]

    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    print(f"Perfect Match GPS (Loss): {cm.GPS():.4f}")  # Returns 0.0000

    # ==============================================================================
    # SCENARIO 2: Verifying the Complement Identity
    # ==============================================================================
    print("\n--- 2. COMPLEMENT IDENTITY CHECK ---")

    cm_bad = ClusteringMetric(y_true=[0, 0, 0], y_pred=[1, 2, 3])

    print(f"Direct GPS:     {cm_bad.GPS()}")
    print(f"1.0 - RaS:      {1.0 - cm_bad.RaS()}")
    print(f"Are they exact? {np.isclose(cm_bad.GPS(), 1.0 - cm_bad.RaS())}")
