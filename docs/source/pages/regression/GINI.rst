GINI - GINI Coefficient
=======================

.. toctree::
   :maxdepth: 3
   :caption: GINI - Gini coefficient

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::



+ Best possible score is 1, bigger value is better. Range = [0, 1]



Latex equation code::



Example to use Gini metric, there are two GINI versions:

.. code-block:: python
	:emphasize-lines: 8-10,16-18

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.gini())
	print(evaluator.gini_wiki())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.GINI(multi_output="raw_values"))
	print(evaluator.GINI_WIKI(multi_output="raw_values"))


