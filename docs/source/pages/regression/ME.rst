ME - Max Error
==============

.. toctree::
   :maxdepth: 3
   :caption: ME - Max Error

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{ME}(y, \hat{y}) = max(| y_i - \hat{y}_i |)

The max_error function computes the maximum residual error , a metric that captures the worst case error between the predicted value and the true value. In a
perfectly fitted single output regression model, max_error would be 0 on the training set and though this would be highly unlikely in the real world, this
metric shows the extent of error that the model had when it was fitted.

+ Best possible score is 0.0, smaller value is better. Range = [0, +inf)


Latex equation code::

	\text{ME}(y, \hat{y}) = max(| y_i - \hat{y}_i |)


Example to use ME metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.max_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.ME(multi_output="raw_values"))
