GAS - Gamma Score
=================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Gamma Score (GAS)** (originally Goodman & Kruskal's Gamma :math:`\gamma`, and widely known in clustering validation as the **Baker-Hubert Gamma Index**) is an external clustering evaluation metric. It measures the rank correlation between two partitions by evaluating the net surplus of concordant sample pairs over discordant pairs, while strictly ignoring tied pairs.

Intuitively, GAS answers the question: *"If we draw two distinct points at random, how much more likely is our model to agree with the ground truth on whether to group or separate them, rather than disagree?"*

.. math::

    \text{GAS} = \frac{S_+ - S_-}{S_+ + S_-}

Where across all :math:`N_d = \binom{N}{2}` possible pairs of distinct data points:

* :math:`S_+ = a + d` (Concordant pairs): Pairs grouped together in both partitions (:math:`a`), plus pairs separated in both partitions (:math:`d`).
* :math:`S_- = b + c` (Discordant pairs): Pairs grouped in ground truth but split in prediction (:math:`b`), plus pairs split in ground truth but grouped in prediction (:math:`c`).

-------------------------------------------------------------------------------

The Mathematical Identity (Relationship with Rand Score)
--------------------------------------------------------

In binary partition comparison, every single non-identical sample pair is classified as either concordant or discordant. Therefore, the denominator strictly equals the total number of pairs: :math:`S_+ + S_- = N_d`. 

By substituting :math:`S_- = N_d - S_+`, the formulation collapses into an exact affine transformation of the standard **Rand Score (RaS)**:

.. math::

    \text{GAS} = \frac{S_+ - (N_d - S_+)}{N_d} = \frac{2S_+ - N_d}{N_d} = 2 \left(\frac{S_+}{N_d}\right) - 1 = 2 \times \text{RaS} - 1

**Clinical/Statistical Takeaway:** While the Rand Score maps to ``[0.0, 1.0]``, the Gamma Score stretches this exact information across ``[-1.0, 1.0]``. Consequently, a completely uninformative, random clustering that yields a Rand Score of ``0.5`` evaluates to a Gamma Score of exactly ``0.0`` (zero correlation baseline).

-------------------------------------------------------------------------------

Algorithmic Optimizations (Performance Note)
--------------------------------------------

Brute-force iteration over all possible sample pairs scales quadratically at :math:`O(N^2)`. 

Leveraging the algebraic identity :math:`\text{GAS} \equiv 2 \cdot \text{RaS} - 1`, this implementation bypasses pairwise combinatorial enumeration entirely. It extracts the exact score directly from the :math:`O(N)` contingency table marginals, ensuring maximum benchmarking velocity.

-------------------------------------------------------------------------------

Handling Edge Cases (Finite Values)
-----------------------------------

The calculation involves division by :math:`S_+ + S_- = N_d`. This denominator can only evaluate to zero if the dataset contains fewer than 2 samples (:math:`N < 2`), meaning no pairwise relationships exist.

* **force_finite (bool):** If ``True``, catches the zero-division error when :math:`N < 2` and returns a safe fallback value instead of raising a ``ZeroDivisionError``. Default is ``True``.
* **finite_value (float):** The fallback value returned when calculation fails. Since a 1-sample dataset trivially represents identical partitions (maximum correlation), the default fallback is ``1.0``.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates identical partitions / 100% concordance).
* **Random baseline:** ``0.0`` (Independent partitions agree no better than coin flips).
* **Worst possible score:** ``-1.0`` (Indicates severe inverse agreement).
* **Permutation Invariance:** Strictly invariant to permutations of cluster labels.
* **Symmetry:** Strictly symmetric: :math:`\text{GAS}(y_{true}, y_{pred}) = \text{GAS}(y_{pred}, y_{true})`.
* **Mathematical Identity:** :math:`\text{GAS} \equiv 2 \cdot \text{RaS} - 1 \equiv \text{Baker-Hubert Gamma}`.
* **Range:** ``[-1.0, 1.0]``
* **References:**

    * `Goodman, Leo A., and William H. Kruskal. "Measures of association for cross classifications." Journal of the American statistical association 49.268 (1954): 732-764. <https://doi.org/10.2307/2281536>`_
    * `Baker, Frank B., and Lawrence J. Hubert. "Measuring the power of hierarchical cluster analysis." Journal of the American Statistical Association 70.349 (1975): 31-38. <https://doi.org/10.2307/2285371>`_

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11,12,20

    from permetrics.clustering import ClusteringMetric

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC GAMMA SCORE EXAMPLE ---")

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 2]
    
    cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)
    gas_score = cm.GAS()
    print(f"Gamma Score: {gas_score:.4f}")

    # ==============================================================================
    # SCENARIO 2: Verifying the Rand Score Identity
    # ==============================================================================
    print("\n--- 2. RAND SCORE AFFINE TRANSFORMATION CHECK ---")

    ras_score = cm.RaS()
    derived_gas = (2 * ras_score) - 1

    print(f"Direct GAS:  {gas_score}")
    print(f"2*RaS - 1:   {derived_gas}")
    print(f"Are they exact? {np.isclose(gas_score, derived_gas)}")
