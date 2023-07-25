AE - Absolute Error
===================

.. toctree::
   :maxdepth: 3
   :caption: AE - Absolute Error

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{AE}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n} | \hat{y}_i - y_i |


+ Best possible score is 0.0, smaller value is better. Range = (-inf, +inf)
+ Computes the absolute error between two numbers, or for element between a pair of list, tuple or numpy arrays.

Latex equation code::

	\text{AE}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n} | \hat{y}_i - y_i |


Example to use AE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.single_absolute_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.AE())

