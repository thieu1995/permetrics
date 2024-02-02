SE - Squared Error
==================

.. toctree::
   :maxdepth: 3
   :caption: SE - Squared Error

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{SE}(y, f_i) = \frac{1}{n}\sum_{i=1}^{n}(y_i - f_i)^2

Latex equation code::

	\text{SE}(y, f_i) = \frac{1}{n}\sum_{i=1}^{n}(y_i - f_i)^2

+ Best possible score is 0.0, smaller value is better. Range = [0, +inf)
+ Note: Computes the squared error between two numbers, or for element between a pair of list, tuple or numpy arrays.
+ The Squared Error (SE) is a metric used to evaluate the accuracy of a regression model by measuring the average of the squared differences between the
predicted and actual values.


Example to use SE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.single_squared_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.SE())
