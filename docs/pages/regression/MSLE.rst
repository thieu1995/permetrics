MSLE - Mean Squared Logarithmic Error
=====================================

.. toctree::
   :maxdepth: 2
   :caption: MSLE - Mean Squared Logarithmic Error


.. math::

    \text{MSLE}(y, \hat{y}) = \frac{1}{N} \sum_{i=0}^{N - 1} (\log_e (1 + y_i) - \log_e (1 + \hat{y}_i) )^2

Where :math:`\log_e (x)` means the natural logarithm of x. This metric is best to use when targets having exponential growth, such as population counts,
average sales of a commodity over a span of years etc. Note that this metric penalizes an under-predicted estimate greater than an over-predicted estimate.

Smaller values are better.


Latex equation code::

	\text{MSLE}(y, \hat{y}) = \frac{1}{N} \sum_{i=0}^{N - 1} (\log_e (1 + y_i) - \log_e (1 + \hat{y}_i) )^2


Example to use: MSLE function::

	from numpy import array
	from permetrics.regression import Metrics

	## 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	obj1 = Metrics(y_true, y_pred)
	print(obj1.mean_squared_log_error(clean=True, decimal=5))

	## > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	multi_outputs = [None, "raw_values", [0.3, 1.2], array([0.5, 0.2]), (0.1, 0.9)]
	obj2 = Metrics(y_true, y_pred)
	for multi_output in multi_outputs:
	    print(obj2.mean_squared_log_error(clean=True, multi_output=multi_output, decimal=5))




