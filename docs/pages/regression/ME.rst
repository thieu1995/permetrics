ME - Max Error
==============

.. toctree::
   :maxdepth: 2
   :caption: ME - Max Error

.. math::

	\text{ME}(y, \hat{y}) = max(| y_i - \hat{y}_i |)

The max_error function computes the maximum residual error , a metric that captures the worst case error between the predicted value and the true value. In a
perfectly fitted single output regression model, max_error would be 0 on the training set and though this would be highly unlikely in the real world, this
metric shows the extent of error that the model had when it was fitted.
+ Smaller values are better.


Latex equation code::

	\text{ME}(y, \hat{y}) = max(| y_i - \hat{y}_i |)


Example to use: ME function::

	from numpy import array
	from permetrics.regression import Metrics

	## 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	obj1 = Metrics(y_true, y_pred)
	print(obj1.max_error(clean=True, decimal=5))

	## > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	multi_outputs = [None, "raw_values", [0.3, 1.2], array([0.5, 0.2]), (0.1, 0.9)]
	obj2 = Metrics(y_true, y_pred)
	for multi_output in multi_outputs:
	    print(obj2.max_error(clean=False, multi_output=multi_output, decimal=5))



