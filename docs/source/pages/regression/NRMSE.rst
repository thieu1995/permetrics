NRMSE - Normalized Root Mean Square Error
=========================================

.. toctree::
   :maxdepth: 3
   :caption: NRMSE - Normalized Root Mean Square Error

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::



+ Normalized Root Mean Square Error (NRMSE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)
+ Link: https://medium.com/microsoftazure/how-to-better-evaluate-the-goodness-of-fit-of-regressions-990dbf1c0091


+ Latex equation code:



+ Example to use NMRSE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.normalized_root_mean_square_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6], [1, 2], [2.1, 2.2], [3.4, 5.5]])
	y_pred = array([[0, 2], [-1, 2], [8, -5], [1.1, 1.9], [2.0, 2.3], [3.0, 4.2]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.NRMSE(multi_output="raw_values"))


