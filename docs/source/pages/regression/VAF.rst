VAF - Variance Accounted For
============================

.. toctree::
   :maxdepth: 3
   :caption: VAF - Variance Accounted For

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{VAF}(y, f_i) = 100\% \times \frac{\sum_{i=1}^{n}(y_i - \bar{y})(f_i - \bar{f})}{\sum_{i=1}^{n}(y_i - \bar{y})^2}



+ Variance Accounted For between 2 signals (VAF): Best possible score is 100% (identical signal), bigger value is better. Range = (-inf, 100%]
+ Link: https://www.dcsc.tudelft.nl/~jwvanwingerden/lti/doc/html/vaf.html
+ Variance Accounted For (VAF) is a metric used to evaluate the performance of a regression model. It measures the proportion of the total variance in the actual values that is accounted for by the variance in the predicted values.


Latex equation code::

	\text{VAF}(y, f_i) = 100\% \times \frac{\sum_{i=1}^{n}(y_i - \bar{y})(f_i - \bar{f})}{\sum_{i=1}^{n}(y_i - \bar{y})^2}


Example to use VAF metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.variance_accounted_for())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.VAF(multi_output="raw_values"))


