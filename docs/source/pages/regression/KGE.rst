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


The Kling-Gupta Efficiency (KGE) :cite:`van2023groundwater` is a statistical measure used to evaluate the performance of hydrological models. It was proposed
to overcome the limitations of other measures such as Nash-Sutcliffe Efficiency and R-squared that focus only on reproducing the mean and variance of
observed data.

The KGE combines three statistical metrics: correlation coefficient, variability ratio and bias ratio, into a single measure of model
performance. The KGE ranges between -infinity and 1, where a value of 1 indicates perfect agreement between the model predictions and the observed data.

The KGE measures not only the accuracy of the model predictions but also its ability to reproduce the variability and timing of the observed data. It has
been widely used in hydrology and water resources engineering to evaluate the performance of hydrological models in simulating streamflow, groundwater
recharge, and water quality parameters.

+ Best possible score is 1, bigger value is better. Range = (-inf, 1]
+ https://rstudio-pubs-static.s3.amazonaws.com/433152_56d00c1e29724829bad5fc4fd8c8ebff.html


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
