MRE - Mean Relative Error
=========================

.. toctree::
   :maxdepth: 3
   :caption: MRE - Mean Relative Error

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{MRE}(y, \hat{y}) = \frac{1}{N} \sum_{i=0}^{N - 1} \frac{|y_i - \hat{y}_i|}{|y_i|}

Latex equation code::

	\text{MRE}(y, \hat{y}) = \frac{1}{N} \sum_{i=0}^{N - 1} \frac{|y_i - \hat{y}_i|}{|y_i|}}


+ Mean Relative Error (MRE) or Mean Relative Bias (MRB)
+ Best possible score is 0.0, smaller value is better. Range = [0, +inf)


Example to use MRE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.mean_relative_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.MRE(multi_output="raw_values"))
