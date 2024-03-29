OI - Overall Index
==================

.. toctree::
   :maxdepth: 3
   :caption: OI - Overall Index


.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

   \text{OI}(y, \hat{y}) = \frac{1}{2} \biggr[ 1 - \frac{RMSE}{y_{max} - y_{min}} + EC \biggr]

Latex equation code::

	\text{OI}(y, \hat{y}) = \frac{1}{2} \biggr[ 1 - \frac{RMSE}{y_{max} - y_{min}} + EC \biggr]


The Overall Index (OI) :cite:`almodfer2022modeling` is a composite measure used to evaluate the accuracy of a forecasting model. It combines the Root Mean
Squared Error (RMSE) with a measure of the relative error and a correction term.
+ Best possible value = 1, bigger value is better. Range = [-1, +1)

Example to use COR metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.overall_index())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.OI(multi_output="raw_values"))
