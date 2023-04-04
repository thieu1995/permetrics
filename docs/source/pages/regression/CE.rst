CE - Cross Entropy
==================

.. toctree::
   :maxdepth: 3
   :caption: CE - Cross Entropy

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

    \text{CE}(y, \hat{y}) = -\frac{1}{n}\sum_{i=1}^{n} \left[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]


+ Range = (-inf, 0]. Can't give comment about this one
+ Greater value of Entropy, the greater the uncertainty for probability distribution and smaller the value the less the uncertainty
+ https://datascience.stackexchange.com/questions/20296/cross-entropy-loss-explanation


Latex equation code::

	\text{CE}(y, \hat{y}) = -\frac{1}{n}\sum_{i=1}^{n} \left[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]


Example to use CE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.cross_entropy())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.CE(multi_output="raw_values"))


