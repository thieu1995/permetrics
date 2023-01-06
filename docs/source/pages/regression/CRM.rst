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


+ Best possible value = 0, smaller value is better. Range = (-inf, +inf)


Latex equation code::

	\text{CRM}(y, \hat{y}) = \frac{\sum{\hat{Y}} - \sum{Y}}{\sum{Y}}

+ https://doi.org/10.1016/j.csite.2022.101797

Example to use CRM metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.coefficient_of_residual_mass())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.CRM(multi_output="raw_values"))

