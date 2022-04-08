EVS - Explained Variance Score
==============================

.. toctree::
   :maxdepth: 3
   :caption: EVS - Explained Variance Score

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


+ The explained variance score computes the `explained variance regression score`_. If Var is Variance, the square of the standard deviation, then the explained variance is estimated as follow:

.. math::

    EVS = 1 - \frac{Var\{ y_{true} - y_{pred} \}}{Var \{ y_{true} \} }

+ Best possible score is 1.0, greater values are better. Range = (-inf, 1.0]


Latex equation code::

    EVS = 1 - \frac{Var\{ y_{true} - y_{pred} \}}{Var \{ y_{true} \} }


Example to use EVS metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.explained_variance_score())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.EVS(multi_output="raw_values"))


.. _explained variance regression score: https://en.wikipedia.org/wiki/Explained_variation

