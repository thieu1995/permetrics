COR - Correlation
=================

.. toctree::
   :maxdepth: 3
   :caption: COR - Correlation


.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

   \text{COR}(y, \hat{y}) = \frac{ COV(y, \hat{y}) }{ std(y) * std(\hat{y})}


+ Best possible value = 1, bigger value is better. Range = [-1, +1)


Latex equation code::

	\text{COR}(y, \hat{y}) = \frac{ COV(y, \hat{y}) }{ std(y) * std(\hat{y})}

+ Measures the strength of the relationship between variables, is the scaled measure of covariance. It is dimensionless.
+ the correlation coefficient is always a pure value and not measured in any units.
+ https://corporatefinanceinstitute.com/resources/data-science/covariance/

Example to use COR metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.correlation())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.COR(multi_output="raw_values"))

