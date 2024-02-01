MAE - Mean Absolute Error
=========================

.. toctree::
   :maxdepth: 3
   :caption: MAE - Mean Absolute Error

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{MAE}(y, \hat{y}) = \frac{ \sum_{i=0}^{N - 1} |y_i - \hat{y}_i| }{N}

Mean Absolute Error (MAE) :cite:`nguyen2018resource` is a statistical measure used to evaluate the accuracy of a forecasting model, such as a regression
model or a time series model. It measures the average magnitude of the errors between the predicted values and the actual values in the units of the response
variable. The MAE is calculated as the average of the absolute differences between the predicted values and the actual values. In other words, it is the mean
of the absolute errors. Best possible score is 0.0, smaller value is better. Range = [0, +inf)

The MAE is a widely used measure of forecast accuracy because it is easy to understand and interpret. A lower MAE indicates better forecast accuracy.
However, like the RMSE, the MAE is not normalized and is dependent on the scale of the response variable, making it difficult to compare the MAE values
across different datasets with different scales.


Example to use MAE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.mean_absolute_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.MAE(multi_output="raw_values"))





