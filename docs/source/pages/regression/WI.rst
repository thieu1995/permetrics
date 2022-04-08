WI - Willmott Index
===================

.. toctree::
   :maxdepth: 3
   :caption: WI - Willmott Index


.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{WI}(y, \hat{y}) = 1 - \frac{ \sum_{i=0}^{N - 1} (\hat{y_i} - y_i)^2 }{ \sum_{i=0}^{N - 1} (|\hat{y_i} - mean(y)| + |y_i - mean(y)|)^2}


+ Willmott Index (WI): Best possible score is 1.0, bigger value is better. Range = [0, 1]
+ Reference evapotranspiration for Londrina, Paraná, Brazil: performance of different estimation methods
+ https://www.researchgate.net/publication/319699360_Reference_evapotranspiration_for_Londrina_Parana_Brazil_performance_of_different_estimation_methods



Latex equation code::

	\text{WI}(y, \hat{y}) = 1 - \frac{ \sum_{i=0}^{N - 1} (\hat{y_i} - y_i)^2 }{ \sum_{i=0}^{N - 1} (|\hat{y_i} - mean(y)| + |y_i - mean(y)|)^2}


Example to use WI metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.willmott_index())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.WI(multi_output="raw_values"))


