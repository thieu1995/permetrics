RAE - Relative Absolute Error
=============================

.. toctree::
   :maxdepth: 3
   :caption: RAE - Relative Absolute Error

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{RAE}(y, \hat{y}) = \frac{\Big[\sum_{i=1}^{n}(\hat{y}_i - y_i)^2\Big]^{1/2}}{\Big[\sum_{i=1}^{n}(y_i)^2\Big]^{1/2}}

Latex equation code::

	\text{RAE}(y, \hat{y}) = \frac{\Big[\sum_{i=1}^{n}(\hat{y}_i - y_i)^2\Big]^{1/2}}{\Big[\sum_{i=1}^{n}(y_i)^2\Big]^{1/2}}

+ Relative Absolute Error (RAE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)
+ `Link to equation <https://www.statisticshowto.com/relative-absolute-error>`_
+ `Link to equation <https://stackoverflow.com/questions/59499222/how-to-make-a-function-of-mae-and-rae-without-using-librarymetrics>`_
+ The Relative Absolute Error (RAE) is a metric used to evaluate the accuracy of a regression model by measuring the ratio of the mean absolute error to the
mean absolute deviation of the actual values.


Example to use RAE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.relative_absolute_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.RAE(multi_output="raw_values"))
