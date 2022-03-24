WI - Willmott Index
===================

.. toctree::
   :maxdepth: 2
   :caption: WI - Willmott Index


.. math::

	\text{WI}(y, \hat{y}) = 1 - \frac{ \sum_{i=0}^{N - 1} (\hat{y_i} - y_i)^2 }{ \sum_{i=0}^{N - 1} (|\hat{y_i} - mean(y)| + |y_i - mean(y)|)^2}

+ 0 < WI < 1. Larger is better


Latex equation code::

	\text{WI}(y, \hat{y}) = 1 - \frac{ \sum_{i=0}^{N - 1} (\hat{y_i} - y_i)^2 }{ \sum_{i=0}^{N - 1} (|\hat{y_i} - mean(y)| + |y_i - mean(y)|)^2}


Example to use: WI function::

	from numpy import array
	from permetrics.regression import Metrics

	## 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	obj1 = Metrics(y_true, y_pred)
	print(obj1.willmott_index(clean=True, decimal=5))

	## > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	multi_outputs = [None, "raw_values", [0.3, 1.2], array([0.5, 0.2]), (0.1, 0.9)]
	obj2 = Metrics(y_true, y_pred)
	for multi_output in multi_outputs:
	    print(obj2.willmott_index(clean=True, multi_output=multi_output, decimal=5))



