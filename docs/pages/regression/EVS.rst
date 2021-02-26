EVS - Explained Variance Score
==============================

.. toctree::
   :maxdepth: 2
   :caption: EVS - Explained Variance Score


The explained variance score computes the `explained variance regression score`_. If Var is Variance, the square of the standard deviation, then the
explained variance is estimated as follow:

.. math::

    EVS = 1 - \frac{Var\{ y_{true} - y_{pred} \}}{Var \{ y_{true} \} }

+ Best possible score is 1.0, lower values are worse.


Latex equation code::

    EVS = 1 - \frac{Var\{ y_{true} - y_{pred} \}}{Var \{ y_{true} \} }


Example to use: EVS function::

	from numpy import array
	from permetrics.regression import Metrics

	## 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	obj1 = Metrics(y_true, y_pred)
	print(obj1.explained_variance_score(clean=True, decimal=5))

	## > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	multi_outputs = [None, "raw_values", [0.3, 1.2], array([0.5, 0.2]), (0.1, 0.9)]
	obj2 = Metrics(y_true, y_pred)
	for multi_output in multi_outputs:
	    print(obj2.explained_variance_score(clean=False, multi_output=multi_output, decimal=5))


.. _explained variance regression score: https://en.wikipedia.org/wiki/Explained_variation

