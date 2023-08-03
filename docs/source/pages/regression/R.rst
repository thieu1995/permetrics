R - Pearson’s Correlation Index
===============================

.. toctree::
   :maxdepth: 3
   :caption: R - Pearson’s Correlation Index

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{R}(y, \hat{y}) = \frac{ \sum_{i=0}^{N - 1} ((y_i - mean(y))*(\hat{y_i} - mean(\hat{y}))) }{ \sqrt{ \sum_{i=0}^{N - 1} (y_i - mean(y))^2} *\sqrt{ \sum_{i=0}^{N - 1} (\hat{y_i} - mean(\hat{y}))^2} }

Pearson’s Correlation Index, also known as Pearson’s correlation coefficient :cite:`van2023groundwater`, is a statistical measure that quantifies the strength
and direction of the linear relationship between two variables. It is denoted by the symbol "r", and ranges between -1 and +1.

A value of +1 indicates a perfect positive linear relationship between the two variables, while a value of -1 indicates a perfect negative linear
relationship. A value of 0 indicates no linear relationship between the two variables.
The Pearson correlation coefficient can be used to determine the strength and direction of the relationship between two variables. A value of r close to +1
indicates a strong positive correlation, while a value close to -1 indicates a strong negative correlation. A value of r close to 0 indicates no correlation.


+ Pearson’s Correlation Coefficient (PCC or R) : Best possible score is 1.0, bigger value is better. Range = [-1, 1]
+ The Pearson correlation coefficient is commonly used in various fields, including social sciences, economics, and engineering, to study the relationship
between two variables.
+ It is important to note that the Pearson correlation coefficient only measures linear relationships between variables, and may not
capture other types of relationships, such as nonlinear or non-monotonic relationships.


Latex equation code::

	\text{R}(y, \hat{y}) = \frac{ \sum_{i=0}^{N - 1} ((y_i - mean(y))*(\hat{y_i} - mean(\hat{y}))) }{ \sqrt{ \sum_{i=0}^{N - 1} (y_i - mean(y))^2} *\sqrt{ \sum_{i=0}^{N - 1} (\hat{y_i} - mean(\hat{y}))^2} }


Example to use R metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.pearson_correlation_coefficient())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.R(multi_output="raw_values"))
