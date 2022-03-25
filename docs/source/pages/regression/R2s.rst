R2s - (Pearson’s Correlation Index)**2
======================================

.. toctree::
   :maxdepth: 3
   :caption: R2s - (Pearson’s Correlation Index)**2

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{R2s}(y, \hat{y}) = \Bigg[ \frac{ \sum_{i=0}^{N - 1} ((y_i - mean(y))*(\hat{y_i} - mean(\hat{y}))) }{ \sqrt{ \sum_{i=0}^{N - 1} (y_i - mean(y))^2}*\sqrt{\sum_{i=0}^{N - 1} (\hat{y_i} - mean(\hat{y}))^2} } \Bigg]^2

+ Best possible score is 1.0, bigger value is better. Range = [0, 1]


Latex equation code::

	\text{R2s}(y, \hat{y}) = \Bigg[ \frac{ \sum_{i=0}^{N - 1} ((y_i - mean(y))*(\hat{y_i} - mean(\hat{y}))) }{ \sqrt{ \sum_{i=0}^{N - 1} (y_i - mean(y))^2}*\sqrt{\sum_{i=0}^{N - 1} (\hat{y_i} - mean(\hat{y}))^2} } \Bigg]^2


Example to use R2s metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.pearson_correlation_coefficient_square())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.R2s(multi_output="raw_values"))

