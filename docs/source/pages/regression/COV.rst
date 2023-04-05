COV - Covariance
================

.. toctree::
   :maxdepth: 3
   :caption: COV - Covariance


.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

   \text{COV}(y, \hat{y}) = \frac{\sum_{i=1}^{N} (y_i - mean(Y)) (\hat{y}_i - mean(\hat{Y}))}{N}


+ There is no best value, bigger value is better. Range = [-inf, +inf)
+ Positive covariance: Indicates that two variables tend to move in the same direction.
+ Negative covariance: Reveals that two variables tend to move in inverse directions.


Latex equation code for covariance of population::

	\text{COV}(y, \hat{y}) = \frac{\sum_{i=1}^{N} (y_i - mean(Y)) (\hat{y}_i - mean(\hat{Y}))}{N}

Latex equation code for covariance of sample::

	\text{COV}(y, \hat{y}) = \frac{\sum_{i=1}^{N} (y_i - mean(Y)) (\hat{y}_i - mean(\hat{Y}))}{N - 1}

+ COV is a measure of the relationship between two random variables.
+ evaluates how much – to what extent – the variables change together.
+ does not assess the dependency between variables.
+ https://corporatefinanceinstitute.com/resources/data-science/covariance/

Example to use COV metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.covariance())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.COV(multi_output="raw_values"))
