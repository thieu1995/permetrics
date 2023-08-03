MSLE - Mean Squared Logarithmic Error
=====================================

.. toctree::
   :maxdepth: 3
   :caption: MSLE - Mean Squared Logarithmic Error

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

    \text{MSLE}(y, \hat{y}) = \frac{1}{N} \sum_{i=0}^{N - 1} (\log_e (1 + y_i) - \log_e (1 + \hat{y}_i) )^2

Where :math:`\log_e (x)` means the natural logarithm of x. This metric is best to use when targets having exponential growth, such as population counts,
average sales of a commodity over a span of years etc. Note that this metric penalizes an under-predicted estimate greater than an over-predicted estimate.

The Mean Squared Logarithmic Error (MSLE) :cite:`hodson2021mean` is a statistical measure used to evaluate the accuracy of a forecasting model, particularly
when the data has a wide range of values. It measures the average of the squared differences between the logarithms of the predicted and actual values.

The logarithmic transformation used in the MSLE reduces the impact of large differences between the actual and predicted values and provides a better
measure of the relative errors between the two values. The MSLE is always a positive value, with a smaller MSLE indicating better forecast accuracy.

+ The MSLE is commonly used in applications such as demand forecasting, stock price prediction, and sales forecasting, where the data has a wide range of
values and the relative errors are more important than the absolute errors.
+ It is important to note that the MSLE is not suitable for data with negative values or zero values, as the logarithm function is not defined for these values.
+ Best possible score is 0.0, smaller value is better. Range = [0, +inf)


Latex equation code::

	\text{MSLE}(y, \hat{y}) = \frac{1}{N} \sum_{i=0}^{N - 1} (\log_e (1 + y_i) - \log_e (1 + \hat{y}_i) )^2


Example to use MSLE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.mean_squared_log_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6], [1, 2]])
	y_pred = array([[0, 2], [-1, 2], [8, -5], [1.1, 1.9]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.MSLE(multi_output="raw_values"))
