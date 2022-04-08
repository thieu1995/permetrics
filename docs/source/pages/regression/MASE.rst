MASE - Mean Absolute Scaled Error
=================================

.. toctree::
   :maxdepth: 3
   :caption: MASE - Mean Absolute Scaled Error

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{MASE}(y, \hat{y}) = \frac{ \frac{1}{N} \sum_{i=0}{N-1} |y_i - \hat{y_i}| }{ \frac{1}{N-1} \sum_{i=1}^{N-1} |y_i - y_{i-1}| }

+ Best possible score is 0.0, smaller value is better. Range = [0, +inf)
+ m = 1 for non-seasonal data, m > 1 for seasonal data
+ `document`_.


Latex equation code::

	\text{MASE}(y, \hat{y}) = \frac{ \frac{1}{N} \sum_{i=0}{N-1} |y_i - \hat{y_i}| }{ \frac{1}{N-1} \sum_{i=1}^{N-1} |y_i - y_{i-1}| }


Example to use MASE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.mean_absolute_scaled_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.MASE(multi_output="raw_values"))

.. _document: https://en.wikipedia.org/wiki/Mean_absolute_scaled_error

