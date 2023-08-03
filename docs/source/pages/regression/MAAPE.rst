MAAPE - Mean Arctangent Absolute Percentage Error
=================================================

.. toctree::
   :maxdepth: 3
   :caption: MAAPE - Mean Arctangent Absolute Percentage Error


.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	MAAPE = \frac{100}{n} \sum_{i=1}^{n} \left|\frac{A_i - F_i}{A_i}\right| \arctan\left(\frac{A_i - F_i}{A_i}\right)

where A_i is the i-th actual value, F_i is the i-th forecasted value, and n is the number of observations.

The Mean Arctangent Absolute Percentage Error (MAAPE) is a statistical measure used to evaluate the accuracy of a forecasting model. It was introduced by
Armstrong in 1985 as an alternative to the Mean Absolute Percentage Error (MAPE) that avoids the issue of dividing by zero when the actual value is zero.

The MAAPE is calculated as the average of the arctangent of the absolute percentage errors between the forecasted and actual values. The arctangent function
is used to transform the percentage errors into a bounded range of -pi/2 to pi/2, which is more suitable for averaging than the unbounded range of the
percentage errors.

The MAAPE measures the average magnitude and direction of the errors between the forecasted and actual values, with values ranging from 0% to 100%. A lower
MAAPE indicates better forecast accuracy. The MAAPE is commonly used in time series forecasting applications, such as sales forecasting, stock price
prediction, and demand forecasting.

+ Best possible score is 0.0, smaller value is better. Range = [0, +inf)
+ https://support.numxl.com/hc/en-us/articles/115001223463-MAAPE-Mean-Arctangent-Absolute-Percentage-Error


Example to use MAAPE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.mean_arctangent_absolute_percentage_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.MAAPE(multi_output="raw_values"))
