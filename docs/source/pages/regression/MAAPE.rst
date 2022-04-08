MAAPE - Mean Arctangent Absolute Percentage Error
=================================================

.. toctree::
   :maxdepth: 3
   :caption: MAAPE - Mean Arctangent Absolute Percentage Error


.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{MAAPE}(y, \hat{y}) = \frac{1}{N} \sum_{i=0}^{N-1} AAPE_i =  \frac{1}{N} \sum_{i=0}^{N - 1} arctan(\frac{|y_i - \hat{y}_i|}{|y_i|})

+ Best possible score is 0.0, smaller value is better. Range = [0, +inf)
+  Link: https://support.numxl.com/hc/en-us/articles/115001223463-MAAPE-Mean-Arctangent-Absolute-Percentage-Error


Latex equation code::

	\text{MAAPE}(y, \hat{y}) = \frac{1}{N} \sum_{i=0}^{N-1} AAPE_i =  \frac{1}{N} \sum_{i=0}^{N - 1} arctan(\frac{|y_i - \hat{y}_i|}{|y_i|})


Example to use MAAPE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.mean_arctangent_absolute_percentage_error())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.MAAPE(multi_output="raw_values"))


.. document: https://support.numxl.com/hc/en-us/articles/115001223463-MAAPE-Mean-Arctangent-Absolute-Percentage-Error

