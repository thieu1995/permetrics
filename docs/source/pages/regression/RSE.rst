RSE - Residual Standard Error
=============================

.. toctree::
   :maxdepth: 3
   :caption: RSE - Residual Standard Error

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{RSE}(y, f_i) = \sqrt{\frac{\sum_{i=1}^{n}(y_i - f_i)^2}{n-p-1}}


+ Residual Standard Error (RSE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)
+ https://www.statology.org/residual-standard-error-r/
+ https://machinelearningmastery.com/degrees-of-freedom-in-machine-learning/
+  The Residual Standard Error (RSE) is a metric used to evaluate the goodness of fit of a regression model. It measures the average distance between the observed values and the predicted values.


Latex equation code::

	\text{RSE}(y, f_i) = \sqrt{\frac{\sum_{i=1}^{n}(y_i - f_i)^2}{n-p-1}}


Example to use RSE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.residual_standard_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.RSE(multi_output="raw_values"))


