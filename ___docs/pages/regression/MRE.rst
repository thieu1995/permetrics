MRE - Mean Relative Error
=========================

.. toctree::
   :maxdepth: 2
   :caption: MRE - Mean Relative Error


.. math::

	\text{MRE}(y, \hat{y}) = \frac{1}{N} \sum_{i=0}^{N - 1} \frac{|y_i - \hat{y}_i|}{|y_i|}

+ Smaller values are better.


Latex equation code::

	\text{MRE}(y, \hat{y}) = \frac{1}{N} \sum_{i=0}^{N - 1} \frac{|y_i - \hat{y}_i|}{|y_i|}}


Example to use: MRE function::

	from numpy import array
	from permetrics.regression import Metrics

	## 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	obj1 = Metrics(y_true, y_pred)
	print(obj1.mean_relative_error(clean=True, decimal=5))

	## > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	multi_outputs = [None, "raw_values", [0.3, 1.2], array([0.5, 0.2]), (0.1, 0.9)]
	obj2 = Metrics(y_true, y_pred)
	for multi_output in multi_outputs:
	    print(obj2.mean_relative_error(clean=True, multi_output=multi_output, decimal=5))




