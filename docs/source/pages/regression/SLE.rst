SLE - Squared Log Error
=======================

.. toctree::
   :maxdepth: 3
   :caption: SLE - Squared Log Error

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{SLE}(y, \hat{y}) =


+ Squared Log Error (SLE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)
+ Note: Computes the squared log error between two numbers, or for element between a pair of list, tuple or numpy arrays.


Latex equation code::

	\text{SLE}(y, \hat{y}) =


Example to use SLE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.single_squared_log_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.SLE())


