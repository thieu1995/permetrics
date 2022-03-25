NSE - Nash-Sutcliffe Efficiency
===============================

.. toctree::
   :maxdepth: 3
   :caption: NSE - Nash-Sutcliffe Efficiency


.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{NSE}(y, \hat{y}) = 1 - \frac{\sum_{i=0}^{N - 1} (y_i - \hat{y_i})^2}{ \sum_{i=0}^{N - 1} (y_i - mean(y))}

+ Best possible score is 1.0, bigger value is better. Range = (-inf, 1]


Latex equation code::

	\text{NSE}(y, \hat{y}) = 1 - \frac{\sum_{i=0}^{N - 1} (y_i - \hat{y_i})^2}{ \sum_{i=0}^{N - 1} (y_i - mean(y))}


Example to use NSE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.nash_sutcliffe_efficiency())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.NSE(multi_output="raw_values"))


