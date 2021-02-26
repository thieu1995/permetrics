CI - Confidence Index
=====================

.. toctree::
   :maxdepth: 2
   :caption: CI - Confidence Index


`Confidence Index`_ (or Performance Index): Reference evapotranspiration for Londrina, Paraná, Brazil: performance of different estimation methods

.. math::

	\text{CI}(y, \hat{y}) = \text{R}(y, \hat{y}) * \text{WI}(y, \hat{y})

Meaning of Values::

	> 0.85          Excellent Model
	0.76-0.85       Very good
	0.66-0.75       Good
	0.61-0.65       Satisfactory
	0.51-0.60       Poor
	0.41-0.50       Bad
	≤ 0.40          Very bad


Latex equation code::

	\text{CI}(y, \hat{y}) = \text{R}(y, \hat{y}) * \text{WI}(y, \hat{y})


Example to use: CI function::

	from numpy import array
	from permetrics.regression import Metrics

	## 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	obj1 = Metrics(y_true, y_pred)
	print(obj1.confidence_index(clean=True, decimal=5))

	## > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	multi_outputs = [None, "raw_values", [0.3, 1.2], array([0.5, 0.2]), (0.1, 0.9)]
	obj2 = Metrics(y_true, y_pred)
	for multi_output in multi_outputs:
	    print(obj2.confidence_index(clean=True, multi_output=multi_output, decimal=5))


.. _Confidence Index: https://www.researchgate.net/publication/319699360_Reference_evapotranspiration_for_Londrina_Parana_Brazil_performance_of_different_estimation_methods

