KGE - Kling-Gupta Efficiency
============================

.. toctree::
   :maxdepth: 3
   :caption: KGE - Kling-Gupta Efficiency

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{KGE}(y, \hat{y}) = 1 - \sqrt{ (r(y, \hat{y}) - 1)^2 + (\beta(y, \hat{y}) - 1)^2 + (\gamma(y, \hat{y}) - 1)^2 }

where: r = correlation coefficient, CV = coefficient of variation, :math:`\mu` = mean, :math:`\sigma` = standard deviation.


.. math::

    \beta = \text{bias ratio} = \frac{\mu_{\hat{y}} }{\mu_{y}}

	\gamma = \text{variability ratio} = \frac{ CV_{\hat{y}} } {CV_y} = \frac{ \sigma _{\hat{y}} / \mu _{\hat{y}} }{ \sigma _y / \mu _y}


+ Best possible score is 1, bigger value is better. Range = (-inf, 1]
+ Link: https://rstudio-pubs-static.s3.amazonaws.com/433152_56d00c1e29724829bad5fc4fd8c8ebff.html


Latex equation code::

	\text{KGE}(y, \hat{y}) = 1 - \sqrt{ (r(y, \hat{y}) - 1)^2 + (\beta(y, \hat{y}) - 1)^2 + (\gamma(y, \hat{y}) - 1)^2 }
	where:
		r = correlation coefficient
		\beta = \text{bias ratio} = \frac{\mu_{\hat{y}} }{\mu_{y}}
		\gamma = \text{variability ratio} = \frac{ CV_{\hat{y}} } {CV_y} = \frac{ \sigma _{\hat{y}} / \mu _{\hat{y}} }{ \sigma _y / \mu _y}
	and:
		CV = coefficient of variation
		\mu = mean
		\sigma = standard deviation


Example to use KGE metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.kling_gupta_efficiency())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.KGE(multi_output="raw_values"))


