AR2 - Adjusted R2
=================

.. toctree::
   :maxdepth: 3
   :caption: AR2 - Adjusted R2

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	AR2(y, \hat{y}) =

where :math:`\bar{y} = \frac{1}{N} \sum_{i=1}^{N} y_i` and :math:`\sum_{i=1}^{N} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{N} \epsilon_i^2`


+ Adjusted Coefficient of Determination (ACOD/AR2): Best possible score is 1.0, bigger value is better. Range = (-inf, 1]
+ https://dziganto.github.io/data%20science/linear%20regression/machine%20learning/python/Linear-Regression-101-Metrics/
+ Scikit-learn and other websites denoted COD as R^2 (or R squared), it leads to the misunderstanding of R^2 in which R is PCC.
+ We should denote it as COD or R2 only.


Latex equation code::

	\text{AR2}(y, \hat{y}) = \textrm{Adjusted R}^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - k - 1}

Here, $n$ is the sample size, $k$ is the number of predictors, $R^2$ is the coefficient of determination, and the Adjusted R2 is calculated as a modification
of the R2 that takes into account the number of predictors in the model. The Adjusted R2 provides a more accurate measure of the goodness-of-fit of a model
with multiple predictors.


Example to use AR2 metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.adjusted_coefficient_of_determination())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.AR2(multi_output="raw_values"))
