AR - Absolute Pearson’s Correlation Index
=========================================

.. toctree::
   :maxdepth: 3
   :caption: AR - Absolute Pearson’s Correlation Index

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{R}(y, \hat{y}) = \frac{ \sum_{i=0}^{N - 1} (|y_i - mean(y)|*|\hat{y_i} - mean(\hat{y})|) }{ \sqrt{ \sum_{i=0}^{N - 1} (y_i - mean(y))^2} *\sqrt{\sum_{i=0}^{N - 1} (\hat{y_i} - mean(\hat{y}))^2} }

Latex equation code::

	\text{AR}(y, \hat{y}) = \frac{ \sum_{i=0}^{N - 1} (|y_i - mean(y)|*|\hat{y_i} - mean(\hat{y})|) }{ \sqrt{ \sum_{i=0}^{N - 1} (y_i - mean(y))^2} *\sqrt{\sum_{i=0}^{N - 1} (\hat{y_i} - mean(\hat{y}))^2} }

+ Absolute Pearson’s Correlation Coefficient (APCC or AR): Best possible score is 1.0, bigger value is better. Range = [0, 1]
+ I developed this method, do not have enough time to analysis this metric.


Example to use AR metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.absolute_pearson_correlation_coefficient())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.AR(multi_output="raw_values"))
