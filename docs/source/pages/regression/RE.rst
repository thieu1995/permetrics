RE - Relative Error
===================

.. toctree::
   :maxdepth: 3
   :caption: RE - Relative Error

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{RE}(y, \hat{y}) = \frac{|y_i - \hat{y}_i|}{|y_i|}


+ Relative Error (RE): Best possible score is 0.0, smaller value is better. Range = (-inf, +inf)
+ Note: Computes the relative error between two numbers, or for element between a pair of list, tuple or numpy arrays.
+ The Relative Error (RE) is a metric used to evaluate the accuracy of a regression model by measuring the ratio of the absolute error to the actual value.


Latex equation code::

	\text{RE}(y, \hat{y}) = \frac{|y_i - \hat{y}_i|}{|y_i|}


Example to use RE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.single_relative_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.RE())
