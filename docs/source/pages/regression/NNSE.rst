NNSE - Normalized NSE
=====================

.. toctree::
   :maxdepth: 3
   :caption: NNSE - Normalized NSE

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{NNSE}(y, \hat{y}) = \frac{1}{2 - NSE}


+ Normalize Nash-Sutcliffe Efficiency (NNSE): Best possible score is 1.0, greater value is better. Range = [0, 1]
+ Link: https://agrimetsoft.com/calculators/Nash%20Sutcliffe%20model%20Efficiency%20coefficient


Latex equation code::

	\text{E}(y, \hat{y}) = \frac{1}{2 - NSE}


Example to use NNSE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.normalized_nash_sutcliffe_efficiency())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6], [1, 2], [2.1, 2.2], [3.4, 5.5]])
	y_pred = array([[0, 2], [-1, 2], [8, -5], [1.1, 1.9], [2.0, 2.3], [3.0, 4.2]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.NNSE(multi_output="raw_values"))


