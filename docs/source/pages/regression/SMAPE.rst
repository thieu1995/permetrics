SMAPE - Symmetric Mean Absolute Percentage Error
================================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

In time series forecasting, the **Symmetric Mean Absolute Percentage Error (SMAPE)** is notorious for having multiple
conflicting definitions across literature. To completely eliminate ambiguity, `permetrics` explicitly
implements **four distinct variants** categorized into two mathematical paradigms:

-------------------------------------------------------------------------------

1. The Original Paradigm (Armstrong, 1985)
------------------------------------------

In the original formulation :cite:`armstrong1985long`, the absolute error is divided by the **mean** of the
absolute actual and predicted values: :math:`(|y_i| + |\hat{y}_i|) / 2`. This factor of :math:`1/2` in the denominator
flips up to become a multiplier of :math:`2` in the numerator.

SMAPE (Original Percentage)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The classic percentage variant utilized in the M3-Competition.

.. math::

    \text{SMAPE}(y, \hat{y}) = \frac{100\%}{N} \sum_{i=1}^{N} \frac{2 \cdot |y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}

* **Range:** ``[0, 200%]``
* **Best possible score:** ``0.0``
* **Function alias:** ``evaluator.SMAPE()`` or ``evaluator.symmetric_mean_absolute_percentage_error()``

SMAPE_NP (Original No-Percentage)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The raw ratio of the original Armstrong formula (omitting the 100% multiplier).

.. math::

    \text{SMAPE\_NP}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} \frac{2 \cdot |y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}

* **Range:** ``[0, 2]``
* **Best possible score:** ``0.0``
* **Function alias:** ``evaluator.SMAPE_NP()``

-------------------------------------------------------------------------------

2. The Simplified Paradigm (Makridakis, 1993)
---------------------------------------------

Proposed by Spyros Makridakis :cite:`makridakis1993accuracy` as an alternative to bound the upper limit naturally to 1.0 (or 100%).
It divides the absolute error directly by the **sum** of the absolutes.

SMAPE_S (Simplified No-Percentage)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The raw decimal variant of the simplified formula.

.. math::

    \text{SMAPE\_S}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} \frac{|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}

* **Range:** ``[0, 1]``
* **Best possible score:** ``0.0``
* **Function alias:** ``evaluator.SMAPE_S()``


SMAPE_S_P (Simplified Percentage)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The 100-scaled percentage variant of the simplified formula.

.. math::

    \text{SMAPE\_S\_P}(y, \hat{y}) = \frac{100\%}{N} \sum_{i=1}^{N} \frac{|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}

* **Range:** ``[0, 100%]``
* **Best possible score:** ``0.0``
* **Function alias:** ``evaluator.SMAPE_S_P()``

-------------------------------------------------------------------------------

Mathematical Disambiguation & Rules
-----------------------------------

1. **Zero-Division Protection:** In all four implementations, if both the actual value and the predicted value at
timestamp :math:`i` are strictly zero (:math:`y_i = 0` and :math:`\hat{y}_i = 0`), the element-wise error is
explicitly evaluated as ``0.0`` to prevent ``NaN`` propagation.

2. **Asymmetry Warning:** Despite the name "Symmetric", all SMAPE variants inherently penalize over-forecasting
(:math:`\hat{y}_i > y_i`) more heavily than under-forecasting (:math:`\hat{y}_i < y_i`).


Example to use SMAPE metrics:

.. code-block:: python
	:emphasize-lines: 8-12,18-19

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.symmetric_mean_absolute_percentage_error())
	print(evaluator.SMAPE_NP())
	print(evaluator.SMAPE_S())
	print(evaluator.SMAPE_S_P())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.SMAPE(multi_output="raw_values"))
