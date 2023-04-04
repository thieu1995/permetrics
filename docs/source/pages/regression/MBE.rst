MBE - Mean Bias Error
=====================

.. toctree::
   :maxdepth: 3
   :caption: MBE - Mean Bias Error

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3



.. math::

	\text{MBE}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n}(f_i - y_i)


+ Mean Bias Error (MBE): Best possible score is 0.0. Range = (-inf, +inf)



Latex equation code::

	\text{MBE}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n}(f_i - y_i)


Example to use MBE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.mean_bias_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.MBE(multi_output="raw_values"))



