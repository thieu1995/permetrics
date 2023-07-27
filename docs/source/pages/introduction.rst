Introduction
############

.. image:: https://img.shields.io/badge/release-1.4.0-yellow.svg?style=svg
    :target: https://github.com/thieu1995/permetrics

.. image:: https://img.shields.io/pypi/wheel/gensim.svg?style=svg
    :target: https://pypi.python.org/pypi/permetrics

.. image:: https://readthedocs.org/projects/permetrics/badge/?version=latest
	:target: https://permetrics.readthedocs.io/en/latest/?badge=latest
	:alt: Documentation Status

.. image:: https://img.shields.io/badge/python-3.6+-orange.svg
    :target: https://www.python.org/downloads/release/python-360

.. image:: https://badge.fury.io/py/permetrics.svg?style=svg
    :target: https://badge.fury.io/py/permetrics

.. image:: https://zenodo.org/badge/280617738.svg?style=svg
	:target: https://zenodo.org/badge/latestdoi/280617738

.. image:: https://pepy.tech/badge/permetrics
   :target: https://pepy.tech/project/permetrics

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=svg
    :target: https://github.com/thieu1995/permetrics/blob/master/LICENSE


PerMetrics is library written in Python, for PERformance METRICS (PerMetrics) of machine learning models.

* The goals of this framework are:
    * Combine all metrics for regression, classification and clustering models
    * Helping users in all field access to metrics as fast as possible
    * Perform Qualitative Analysis of models.
    * Perform Quantitative Analysis of models.

* Currently, it contains 2 sub-packages including:
	* regression: contains 47 metrics
	* classification: contains 17 metrics


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

Setup
#####

Install the [current PyPI release](https://pypi.python.org/pypi/permetrics):

This is a simple example::

	pip install permetrics==1.4.0

Or install the development version from GitHub::

	pip install git+https://github.com/thieu1995/permetrics


Examples
########

.. include:: examples/functional_style.rst
.. include:: examples/oop_style.rst
.. include:: examples/multiple_metrics.rst
.. include:: examples/multiple_outputs_multiple_metrics.rst



Important links
###############

* Official source code repo: https://github.com/thieu1995/permetrics
* Official document: https://permetrics.readthedocs.io/
* Download releases: https://pypi.org/project/permetrics/
* Issue tracker: https://github.com/thieu1995/permetrics/issues

* This project also related to my another projects which are "meta-heuristics" and "neural-network", check it here
	* https://github.com/thieu1995/mealpy
	* https://github.com/thieu1995/metaheuristics
    * https://github.com/thieu1995/opfunu
    * https://github.com/chasebk


.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3
