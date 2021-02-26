DRV - Deviation of Runoff Volume
================================

.. toctree::
   :maxdepth: 2
   :caption: DRV - Deviation of Runoff Volume


.. math::

	\text{DRV}(y, \hat{y}) = \frac{ \sum_{i=0}^{N - 1} y_i }{ \sum_{i=0}^{N - 1} \hat{y_i} }

+ Smaller values are better. Best value is 0.


Latex equation code::

	\text{DRV}(y, \hat{y}) = \frac{ \sum_{i=0}^{N - 1} y_i }{ \sum_{i=0}^{N - 1} \hat{y_i} }


Example to use: DRV function::

	from numpy import array
	from permetrics.regression import Metrics

	## 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	obj1 = Metrics(y_true, y_pred)
	print(obj1.deviation_of_runoff_volume(clean=True, decimal=5))

	## > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	multi_outputs = [None, "raw_values", [0.3, 1.2], array([0.5, 0.2]), (0.1, 0.9)]
	obj2 = Metrics(y_true, y_pred)
	for multi_output in multi_outputs:
	    print(obj2.deviation_of_runoff_volume(clean=False, multi_output=multi_output, decimal=5))






