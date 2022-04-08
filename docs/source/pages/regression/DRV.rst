DRV - Deviation of Runoff Volume
================================

.. toctree::
   :maxdepth: 3
   :caption: DRV - Deviation of Runoff Volume

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{DRV}(y, \hat{y}) = \frac{ \sum_{i=0}^{N - 1} y_i }{ \sum_{i=0}^{N - 1} \hat{y_i} }

+ Best possible score is 1.0, smaller value is better. Range = [1, +inf)
+ Link: https://rstudio-pubs-static.s3.amazonaws.com/433152_56d00c1e29724829bad5fc4fd8c8ebff.html


Latex equation code::

	\text{DRV}(y, \hat{y}) = \frac{ \sum_{i=0}^{N - 1} y_i }{ \sum_{i=0}^{N - 1} \hat{y_i} }


Example to use DRV metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.deviation_of_runoff_volume())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.DRV(multi_output="raw_values"))


