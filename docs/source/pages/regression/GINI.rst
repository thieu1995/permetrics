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


The Gini coefficient :cite:`gupta2022integrated` is a statistical measure used to measure income or wealth inequality within a population. It is named
after the Italian statistician Corrado Gini who developed the concept in 1912. The Gini coefficient ranges from 0 to 1, where 0 represents perfect equality
(every individual in the population has the same income or wealth) and 1 represents perfect inequality (one individual has all the income or wealth and
everyone else has none).

.. math::

	G = \frac{A}{A + B}

where G is the Gini coefficient, A is the area between the Lorenz curve and the line of perfect equality, and B is the area under the line of perfect equality.

The Gini coefficient is calculated by plotting the cumulative share of income or wealth (on the x-axis) against the cumulative share of the population (on
the y-axis) and measuring the area between this curve and the line of perfect equality (which is a straight diagonal line from the origin to the upper right
corner of the plot).

The Gini coefficient is widely used in social sciences and economics to measure income or wealth inequality within and between countries. It is also used to
analyze the distribution of other variables, such as educational attainment, health outcomes, and access to resources.

+ Best possible score is 1, bigger value is better. Range = [0, 1]
+ This version is based on: https://github.com/benhamner/Metrics/blob/master/MATLAB/metrics/gini.m



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


