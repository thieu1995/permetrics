Introduction
############

.. image:: https://img.shields.io/badge/release-1.0.4-yellow.svg?style=svg
    :target: https://github.com/thieu1995/permetrics

.. image:: https://img.shields.io/pypi/wheel/gensim.svg?style=svg
    :target: https://pypi.python.org/pypi/permetrics

.. image:: https://readthedocs.org/projects/permetrics/badge/?version=latest
	:target: https://permetrics.readthedocs.io/en/latest/?badge=latest
	:alt: Documentation Status

.. image:: https://img.shields.io/badge/python-3.7+-orange.svg
    :target: https://www.python.org/downloads/release/python-370

.. image:: https://badge.fury.io/py/permetrics.svg?style=svg
    :target: https://badge.fury.io/py/permetrics

.. image:: https://zenodo.org/badge/280617738.svg?style=svg
	:target: https://zenodo.org/badge/latestdoi/280617738

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=svg
    :target: https://github.com/thieu1995/permetrics/blob/master/LICENSE



PerMetrics is library written in Python, for PERformance METRICS (PerMetrics) of machine learning models.

* The goals of this framework are:
    * Combine all metrics for regression, classification and clustering models
    * Helping users in all field access to metrics as fast as possible
    * Perform Qualitative Analysis of models.
    * Perform Quantitative Analysis of models.

* Currently, It contains 3 sub-packages including:
	* regression: contains 21 metrics
	* single loss: contains 5 metrics
	* classification: contains 1 metrics


If you see my code and data useful and use it, please cites my works here::

	@software{thieu_nguyen_2020_3951205,
	  author       = {Thieu Nguyen},
	  title        = {A framework of PERformance METRICS (PerMetrics) for artificial intelligence models},
	  month        = jul,
	  year         = 2020,
	  publisher    = {Zenodo},
	  doi          = {10.5281/zenodo.3951205},
	  url          = {https://doi.org/10.5281/zenodo.3951205}
	}

	@article{nguyen2019efficient,
	  title={Efficient Time-Series Forecasting Using Neural Network and Opposition-Based Coral Reefs Optimization},
	  author={Nguyen, Thieu and Nguyen, Tu and Nguyen, Binh Minh and Nguyen, Giang},
	  journal={International Journal of Computational Intelligence Systems},
	  volume={12},
	  number={2},
	  pages={1144--1161},
	  year={2019},
	  publisher={Atlantis Press}
	}


Setup
#####

Install the [current PyPI release](https://pypi.python.org/pypi/permetrics):

This is a simple example::

	pip install permetrics

Or install the development version from GitHub::

	pip install git+https://github.com/thieu1995/permetrics


Examples
########

+ All you need to do is: (Make sure your y_true and y_pred is a numpy array).

.. code-block:: python
	:emphasize-lines: 9,18

	from numpy import array
	from permetrics.regression import Metrics

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	obj1 = Metrics(y_true, y_pred)
	print(obj1.root_mean_squared_error(clean=True, decimal=5))

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	multi_outputs = [None, "raw_values", [0.3, 1.2], array([0.5, 0.2]), (0.1, 0.9)]
	obj2 = Metrics(y_true, y_pred)
	for multi_output in multi_outputs:
		print(obj2.root_mean_squared_error(clean=False, multi_output=multi_output, decimal=5))
	...



Important links
###############

* Official source code repo: https://github.com/thieu1995/permetrics
* Download releases: https://pypi.org/project/permetrics/
* Issue tracker: https://github.com/thieu1995/permetrics/issues

* This project also related to my another projects which are "meta-heuristics" and "neural-network", check it here
    * https://github.com/thieu1995/opfunu
    * https://github.com/thieu1995/metaheuristics
    * https://github.com/chasebk
