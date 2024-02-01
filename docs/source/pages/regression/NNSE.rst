NNSE - Normalized NSE
=====================

.. toctree::
   :maxdepth: 3
   :caption: NNSE - Normalized NSE

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{NNSE}(y, \hat{y}) = \frac{1}{2 - NSE}

Latex equation code::

	\text{E}(y, \hat{y}) = \frac{1}{2 - NSE}


The Normalized NSE (NNSE) :cite:`ahmed2021comprehensive` is a statistical measure used to evaluate the performance of hydrological models in simulating
streamflow. It is a variant of the Nash-Sutcliffe Efficiency (NSE), which is a widely used measure of model performance in hydrology.

The NNSE accounts for the variability in the observed streamflow and provides a more objective measure of model performance than the NSE alone. The NNSE is
commonly used in hydrology and water resources engineering to evaluate the performance of hydrological models in simulating streamflow and to compare the
performance of different models.

+ Normalize Nash-Sutcliffe Efficiency (NNSE): Best possible score is 1.0, greater value is better. Range = [0, 1]
+ `Link text <https://agrimetsoft.com/calculators/Nash%20Sutcliffe%20model%20Efficiency%20coefficient>`_


Example to use NNSE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.normalized_nash_sutcliffe_efficiency())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6], [1, 2], [2.1, 2.2], [3.4, 5.5]])
	y_pred = array([[0, 2], [-1, 2], [8, -5], [1.1, 1.9], [2.0, 2.3], [3.0, 4.2]])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.NNSE(multi_output="raw_values"))
