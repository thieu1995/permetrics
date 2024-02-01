MBE - Mean Bias Error
=====================

.. toctree::
   :maxdepth: 3
   :caption: MBE - Mean Bias Error

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3



.. math::

	\text{MBE}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n}(f_i - y_i)

Latex equation code::

	\text{MBE}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n}(f_i - y_i)


The Mean Bias Error (MBE) :cite:`kato2016prediction` is a statistical measure used to assess the bias of a forecasting model. The MBE measures the average
difference between the forecasted and actual values, without considering their direction.

The MBE is expressed in the same units as the forecasted and actual values, and a best possible score of 0.0 indicates no bias in the forecasting model. The
MBE has a range of (-infinity, +infinity), with a positive MBE indicating that the forecasted values are, on average, larger than the actual values, and a
negative MBE indicating the opposite.

The MBE is a useful measure to evaluate the systematic errors of a forecasting model, such as overestimation or underestimation of the forecasted values.
However, it does not provide information about the magnitude or direction of the individual errors, and it should be used in conjunction with other
statistical measures, such as the Mean Absolute Error (MAE), to provide a more comprehensive evaluation of the forecasting model's accuracy.

It is important to note that the MBE is sensitive to outliers and may not be appropriate for data with non-normal distributions or extreme values. In such
cases, other measures, such as the Median Bias Error (MBE), may be more appropriate.


Example to use MBE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.mean_bias_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.MBE(multi_output="raw_values"))
