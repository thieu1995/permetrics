CI - Confidence Index
=====================

.. toctree::
   :maxdepth: 3
   :caption: CI - Confidence Index

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{CI}(y, \hat{y}) = \text{R}(y, \hat{y}) * \text{WI}(y, \hat{y})

Confidence Index :cite:`ahmed2021comprehensive` or Performance Index (CI/PI) is score that measures the performance of each estimation method, with a higher value
indicating better performance. The range of the CI/PI is (-inf, 1], meaning it can take any value less than or equal to 1, but not including negative infinity.

+ Best possible score is 1.0, bigger value is better. Range = (-inf, 1], meaning of values::

	> 0.85          Excellent Model
	0.76-0.85       Very good
	0.66-0.75       Good
	0.61-0.65       Satisfactory
	0.51-0.60       Poor
	0.41-0.50       Bad
	< 0.40          Very bad


Latex equation code::

	\text{CI}(y, \hat{y}) = \text{R}(y, \hat{y}) * \text{WI}(y, \hat{y})


Example to use CI metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.confidence_index())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.CI(multi_output="raw_values"))
