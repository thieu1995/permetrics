RMSE - Root Mean Square Error
=============================

.. toctree::
   :maxdepth: 3
   :caption: RMSE - Root Mean Square Error


.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{RMSE}(y, \hat{y}) = \sqrt{\frac{\sum_{i=0}^{N - 1} (y_i - \hat{y}_i)^2}{N}}

Root Mean Square Error (RMSE) :cite:`nguyen2019building` is a statistical measure that is often used to evaluate the accuracy of a forecasting model, such as a
regression model or a time series model. It measures the difference between the predicted values and the actual values in the units of the response variable.

The RMSE is calculated as the square root of the average of the squared differences between the predicted values and the actual values. In other words, it is
the square root of the mean of the squared errors. Best possible score is 0.0, smaller value is better. Range = [0, +inf)

The RMSE is a widely used measure of forecast accuracy because it is sensitive to both the magnitude and direction of the errors. A lower RMSE indicates
better forecast accuracy. However, it has a drawback that it is not normalized, meaning that it is dependent on the scale of the response variable.
Therefore, it is difficult to compare the RMSE values across different datasets with different scales.

The RMSE is commonly used in various fields, including finance, economics, and engineering, to evaluate the performance of forecasting models. It is often
used in conjunction with other measures, such as the Mean Absolute Error (MAE) and the Mean Absolute Percentage Error (MAPE), to provide a more comprehensive
evaluation of the model's performance.


Latex equation code::

	\text{RMSE}(y, \hat{y}) = \sqrt{\frac{\sum_{i=0}^{N - 1} (y_i - \hat{y}_i)^2}{N}}


Example to use RMSE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.root_mean_squared_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.RMSE(multi_output="raw_values"))
