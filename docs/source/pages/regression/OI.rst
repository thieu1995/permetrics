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


+ Best possible value = 1, bigger value is better. Range = [-1, +1)


Latex equation code::

	\text{OI}(y, \hat{y}) = \frac{1}{2} \biggr[ 1 - \frac{RMSE}{y_{max} - y_{min}} + EC \biggr]

+ https://doi.org/10.1016/j.csite.2022.101797

Example to use COR metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.overall_index())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.OI(multi_output="raw_values"))

