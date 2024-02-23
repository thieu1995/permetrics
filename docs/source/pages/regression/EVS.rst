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


.. math::

    EVS = 1 - \frac{Var\{ y_{true} - y_{pred} \}}{Var \{ y_{true} \} }

+ `Link to equation <https://www.oreilly.com/library/view/mastering-python-for/9781789346466/d1ac368a-6890-45eb-b39c-2fa97d23d640.xhtml>`_

The given math formula defines the explained variance score (EVS) :cite:`nguyen2020eo`, which is a metric used in regression analysis to evaluate the
performance of a model. The formula computes the ratio of the variance of the difference between the true values y_true and the predicted values y_pred to
the variance of the true values y_true.

The resulting score ranges between -infinity and 1, with a score of 1 indicating a perfect match between the true and predicted values and a score of 0
indicating that the model does not perform better than predicting the mean of the true values.

A higher value of EVS indicates a better performance of the model. Best possible score is 1.0, greater values are better. Range = (-inf, 1.0].


Example to use EVS metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.explained_variance_score())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.EVS(multi_output="raw_values"))
