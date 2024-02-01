MAPE - Mean Absolute Percentage Error
=====================================

.. toctree::
   :maxdepth: 3
   :caption: MAPE - Mean Absolute Percentage Error

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{MAPE}(y, \hat{y}) = \frac{100\%}{N} \sum_{i=0}^{N - 1} \frac{|y_i - \hat{y}_i|}{|y_i|}

The Mean Absolute Percentage Error (MAPE) :cite:`nguyen2020new` is a statistical measure of the accuracy of a forecasting model, commonly used in
business and economics. The MAPE measures the average percentage difference between the forecasted and actual values, with a lower MAPE indicating better
forecast accuracy.

The MAPE is expressed as a percentage, and a commonly used benchmark for a good forecast model is a MAPE of less than 20%. However, the benchmark may vary
depending on the specific application and industry. The MAPE has a range of [0, +infinity), with a best possible score of 0.0, indicating perfect forecast
accuracy. A larger MAPE indicates a larger average percentage difference between the forecasted and actual values, with infinite MAPE indicating a complete
failure of the forecasting model.

+ `Link equation <https://ibf.org/knowledge/glossary/mape-mean-absolute-percentage-error-174>`_

Example to use MAPE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.mean_absolute_percentage_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.MAPE(multi_output="raw_values"))
