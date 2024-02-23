
<p align="center">
<img style="max-width:100%;" 
src="https://thieu1995.github.io/post/2023-08/permetrics-01.png" 
alt="PERMETRICS"/>
</p>


---

[![GitHub release](https://img.shields.io/badge/release-1.5.0-yellow.svg)](https://github.com/thieu1995/permetrics/releases)
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/permetrics) 
[![PyPI version](https://badge.fury.io/py/permetrics.svg)](https://badge.fury.io/py/permetrics)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/permetrics.svg)
![PyPI - Status](https://img.shields.io/pypi/status/permetrics.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/permetrics.svg)
[![Downloads](https://static.pepy.tech/badge/permetrics)](https://pepy.tech/project/permetrics)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/permetrics/actions/workflows/publish-package.yaml/badge.svg)](https://github.com/thieu1995/permetrics/actions/workflows/publish-package.yaml)
![GitHub Release Date](https://img.shields.io/github/release-date/thieu1995/permetrics.svg)
[![Documentation Status](https://readthedocs.org/projects/permetrics/badge/?version=latest)](https://permetrics.readthedocs.io/en/latest/?badge=latest)
[![Chat](https://img.shields.io/badge/Chat-on%20Telegram-blue)](https://t.me/+fRVCJGuGJg1mNDg1)
![GitHub contributors](https://img.shields.io/github/contributors/thieu1995/permetrics.svg)
[![GitTutorial](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)](https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project)
[![DOI](https://zenodo.org/badge/280617738.svg)](https://zenodo.org/badge/latestdoi/280617738)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


PerMetrics is a python library for performance metrics of machine learning models. We aim to implement all 
performance metrics for problems such as regression, classification, clustering, ... problems. Helping users in all 
field access metrics as fast as possible. The number of available metrics include **111 (47 regression metrics, 20 classification metrics, 44 clustering 
metrics)**


# Citation Request 

Please include these citations if you plan to use this library:

```code 
@software{nguyen_van_thieu_2023_8220489,
  author       = {Nguyen Van Thieu},
  title        = {PerMetrics: A Framework of Performance Metrics for Machine Learning Models},
  month        = aug,
  year         = 2023,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3951205},
  url          = {https://github.com/thieu1995/permetrics}
}
```


# Installation

Install the [current PyPI release](https://pypi.python.org/pypi/permetrics):
```sh 
$ pip install permetrics
```

After installation, you can import Permetrics as any other Python module:

```sh
$ python
>>> import permetrics
>>> permetrics.__version__
```

# Example

Below is the most efficient and effective way to use this library compared to other libraries. 
The example below returns the values of metrics such as root mean squared error, mean absolute error...

```python
from permetrics import RegressionMetric

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

evaluator = RegressionMetric(y_true, y_pred)
results = evaluator.get_metrics_by_list_names(["RMSE", "MAE", "MAPE", "R2", "NSE", "KGE"])
print(results["RMSE"])
print(results["KGE"])
```

In case your y_true and y_pred data have multiple columns, and you want to return multiple outputs, something that other libraries cannot do, you can do it in Permetrics as follows:


```python
import numpy as np
from permetrics import RegressionMetric

y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
y_pred = np.array([[0, 2], [-1, 2], [8, -5]])

evaluator = RegressionMetric(y_true, y_pred)

## The 1st way
results = evaluator.get_metrics_by_dict({
  "RMSE": {"multi_output": "raw_values"},
  "MAE": {"multi_output": "raw_values"},
  "MAPE": {"multi_output": "raw_values"},
})

## The 2nd way
results = evaluator.get_metrics_by_list_names(
  list_metric_names=["RMSE", "MAE", "MAPE", "R2", "NSE", "KGE"],
  list_paras=[{"multi_output": "raw_values"},] * 6
)

## The 3rd way
result01 = evaluator.RMSE(multi_output="raw_values")
result02 = evaluator.MAE(multi_output="raw_values")
```

The more complicated cases in the folder: [examples](/examples). You can also read the [documentation](https://permetrics.readthedocs.io/) 
for more detailed installation instructions, explanations, and examples.


# Contributing

There are lots of ways how you can contribute to Permetrics's development, and you are welcome to join in! For example, 
you can report problems or make feature requests on the [issues](/issues) pages. To facilitate contributions, 
please check for the guidelines in the [CONTRIBUTING.md](/CONTRIBUTING.md) file.


# Official channels 

* [Official source code repository](https://github.com/thieu1995/permetrics)
* [Official document](https://permetrics.readthedocs.io/)
* [Download releases](https://pypi.org/project/permetrics/) 
* [Issue tracker](https://github.com/thieu1995/permetrics/issues) 
* [Notable changes log](/ChangeLog.md)
* [Official discussion group](https://t.me/+fRVCJGuGJg1mNDg1) 


# Warning

* **Currently, there is a huge misunderstanding among frameworks around the world about the notation of R, R2, and R^2.** 
* Please read the file [R-R2-Rsquared.docx](https://github.com/thieu1995/permetrics/blob/master/R-R2-Rsquared.docx) to understand the differences between them and why there is such confusion.

<p align="center"><img src=".github/assets/rr2.png" alt="R" title="R"/></p>

* **My recommendation is to denote the Coefficient of Determination as COD or R2, while the squared Pearson's 
  Correlation Coefficient should be denoted as R^2 or RSQ (as in Excel software).**


---

Developed by: [Thieu](mailto:nguyenthieu2102@gmail.com?Subject=Permetrics_QUESTIONS) @ 2023
