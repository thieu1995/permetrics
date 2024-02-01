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

Latex equation code::

	\text{WI}(y, \hat{y}) = 1 - \frac{ \sum_{i=0}^{N - 1} (\hat{y_i} - y_i)^2 }{ \sum_{i=0}^{N - 1} (|\hat{y_i} - mean(y)| + |y_i - mean(y)|)^2}


The Willmott Index (WI) :cite:`da2017reference` is a statistical measure used to evaluate the performance of a forecasting model, particularly in the
context of hydrological or climate-related variables. The WI compares the accuracy of a model to the accuracy of a reference model that simply predicts the
mean value of the observed variable. Best possible score is 1.0, bigger value is better. Range = [0, 1]

The WI ranges between 0 and 1, with a value of 1 indicating perfect agreement between the predicted and observed values. A value of 0 indicates that the
predicted values are no better than predicting the mean of the observed values.

The WI is commonly used in hydrology and climate-related fields to evaluate the accuracy of models that predict variables such as precipitation, temperature,
and evapotranspiration. It is a useful tool for comparing the performance of different models or different methods of estimating a variable.

+ `Link to equation <https://www.researchgate.net/publication/319699360_Reference_evapotranspiration_for_Londrina_Parana_Brazil_performance_of_different_estimation_methods>`_


Example to use WI metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.willmott_index())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.WI(multi_output="raw_values"))
