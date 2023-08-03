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

Correlation :cite:`joreskog1978structural` measures the strength of the relationship between the variables and is a scaled measure of covariance. The
correlation coefficient ranges from -1 to +1, where a value of 1 indicates a perfect positive correlation, a value of -1 indicates a perfect negative
correlation, and a value of 0 indicates no correlation.

To calculate the correlation coefficient, you divide the covariance of the variables by the product of their standard deviations. This normalization allows
for comparison between different pairs of variables. The correlation coefficient is dimensionless and does not have any specific units of measurement.

+ Best possible value = 1, bigger value is better. Range = [-1, +1)
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
