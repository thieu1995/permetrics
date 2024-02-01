SMAPE - Symmetric Mean Absolute Percentage Error
================================================

.. toctree::
   :maxdepth: 3
   :caption: SMAPE - Symmetric Mean Absolute Percentage Error

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{SMAPE}(y, \hat{y}) = \frac{100\%}{N} \sum_{i=0}^{N - 1} \frac{ 2*|y_i - \hat{y}_i|}{|y| + |\hat{y}|}

Latex equation code::

	\text{SMAPE}(y, \hat{y}) = \frac{100\%}{N} \sum_{i=0}^{N - 1} \frac{ 2*|y_i - \hat{y}_i|}{|y| + |\hat{y}|}


Symmetric Mean Absolute Percentage Error (SMAPE) :cite:`thieu2019efficient`, which is an accuracy measure commonly used in forecasting and time series
analysis.

Given the actual values y and the predicted values y_hat, the SMAPE is calculated as the average of the absolute percentage errors between the two, where
each error is weighted by the sum of the absolute values of the actual and predicted values.

The resulting score ranges between 0 and 1, where a score of 0 indicates a perfect match between the actual and predicted values, and a score of 1 indicates
no match at all. A smaller value of SMAPE is better, and it is often multiplied by 100% to obtain the percentage error. Best possible score is 0.0, smaller
value is better. Range = [0, 1].

+ `Link to equation <https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error>`_


Example to use SMAPE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.symmetric_mean_absolute_percentage_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.SMAPE(multi_output="raw_values"))
