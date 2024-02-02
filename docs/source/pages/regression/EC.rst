EC - Efficiency Coefficient
===========================

.. toctree::
   :maxdepth: 3
   :caption: EC - Efficiency Coefficient


.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

   \text{EC}(y, \hat{y}) = 1 - \frac{ \sum_{i=1}^n (y_i - \hat{y_i})^2 }{ \sum_{i=1}^n (y_i - mean(Y))^2 }

Latex equation code::

	\text{EC}(y, \hat{y}) = 1 - \frac{ \sum_{i=1}^n (y_i - \hat{y_i})^2 }{ \sum_{i=1}^n (y_i - mean(Y))^2 }

Efficiency Coefficient (EC) :cite:`joreskog1978structural` is a metric used to evaluate the accuracy of a regression model in predicting continuous values.

+ Best possible value = 1, bigger value is better. Range = [-inf, +1]
+ The EC ranges from negative infinity to 1, where a value of 1 indicates a perfect match between the model predictions and the observed data, and a value
of 0 indicates that the model predictions are no better than the benchmark prediction.
+ A negative value indicates that the model predictions are worse than the benchmark prediction.
+ `Link to equation <https://doi.org/10.1016/j.csite.2022.101797>`_

Example to use EC metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.efficiency_coefficient())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.EC(multi_output="raw_values"))
