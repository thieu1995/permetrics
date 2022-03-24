KGE - Kling-Gupta Efficiency
============================

.. toctree::
   :maxdepth: 2
   :caption: KGE - Kling-Gupta Efficiency


.. math::

	\text{KGE}(y, \hat{y}) = 1 - \sqrt{ (r(y, \hat{y}) - 1)^2 + (\beta(y, \hat{y}) - 1)^2 + (\gamma(y, \hat{y}) - 1)^2 }

where: r = correlation coefficient, CV = coefficient of variation, :math:`\mu` = mean, :math:`\sigma` = standard deviation.

.. math::

    \beta = \text{bias ratio} = \frac{\mu_{\hat{y}} }{\mu_{y}}

	\gamma = \text{variability ratio} = \frac{ CV_{\hat{y}} } {CV_y} = \frac{ \sigma _{\hat{y}} / \mu _{\hat{y}} }{ \sigma _y / \mu _y}




+ -unlimited < KGE < 1.   Larger is better


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


Example to use: KGE function::

	from numpy import array
	from permetrics.regression import Metrics

	## 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	obj1 = Metrics(y_true, y_pred)
	print(obj1.kling_gupta_efficiency(clean=True, decimal=5))

	## > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	multi_outputs = [None, "raw_values", [0.3, 1.2], array([0.5, 0.2]), (0.1, 0.9)]
	obj2 = Metrics(y_true, y_pred)
	for multi_output in multi_outputs:
	    print(obj2.kling_gupta_efficiency(clean=True, multi_output=multi_output, decimal=5))




