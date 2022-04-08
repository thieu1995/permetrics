RAE - Relative Absolute Error
=============================

.. toctree::
   :maxdepth: 3
   :caption: RAE - Relative Absolute Error

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{RAE}(y, \hat{y}) =


+ Relative Absolute Error (RAE): Best possible score is 0.0, smaller value is better. Range = [0, +inf)
+ https://stackoverflow.com/questions/59499222/how-to-make-a-function-of-mae-and-rae-without-using-librarymetrics
+ https://www.statisticshowto.com/relative-absolute-error


Latex equation code::

	\text{RAE}(y, \hat{y}) =


Example to use RAE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.relative_absolute_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.RAE(multi_output="raw_values"))


