DRV - Deviation of Runoff Volume
================================

.. toctree::
   :maxdepth: 3
   :caption: DRV - Deviation of Runoff Volume

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{DRV}(y, \hat{y}) = \frac{ \sum_{i=0}^{N - 1} y_i }{ \sum_{i=0}^{N - 1} \hat{y_i} }

+ Best possible score is 0, smaller value is better. Range = (-inf, +inf)


Latex equation code::

	\text{DRV}(y, \hat{y}) = \frac{ \sum_{i=0}^{N - 1} y_i }{ \sum_{i=0}^{N - 1} \hat{y_i} }


Example to use DRV metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.deviation_of_runoff_volume())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.DRV(multi_output="raw_values"))


