MAPE - Mean Percentage Error
============================

.. toctree::
   :maxdepth: 3
   :caption: MAPE - Mean Percentage Error

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{MPE}(y, \hat{y}) = \frac{100\%}{N} \sum_{i=0}^{N - 1} \frac{y_i - \hat{y}_i}{y_i}.


+ Mean Percentage Error (MPE): Best possible score is 0.0. Range = (-inf, +inf)
+ Link: https://www.dataquest.io/blog/understanding-regression-error-metrics/


Latex equation code::

	\text{MPE}(y, \hat{y}) = \frac{100\%}{N} \sum_{i=0}^{N - 1} \frac{y_i - \hat{y}_i}{y_i}.


Example to use MPE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.mean_percentage_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.MPE(multi_output="raw_values"))


