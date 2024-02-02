CRM - Coefficient of Residual Mass
==================================

.. toctree::
   :maxdepth: 3
   :caption: CRM - Coefficient of Residual Mass


.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

   \text{CRM}(y, \hat{y}) = \frac{\sum{\hat{Y}} - \sum{Y}}{\sum{Y}}


The CRM :cite:`almodfer2022modeling` is a measure of the accuracy of the model in predicting the values of the dependent variable. A lower value of CRM
indicates that the model is better at predicting the values of the dependent variable, while a higher value indicates poorer performance. The coefficient of
residual mass is typically used in environmental engineering and hydrology to measure the accuracy of models used to predict water quality and quantity,
sediment transport, and erosion.
+ Best possible value = 0, smaller value is better. Range = (-inf, +inf)
+ `Link to equation <https://doi.org/10.1016/j.csite.2022.101797>`_


Example to use CRM metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.coefficient_of_residual_mass())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.CRM(multi_output="raw_values"))
