# Optimization Function in Numpy (OpFuNu)
[![GitHub release](https://img.shields.io/badge/release-1.0.1-yellow.svg)]()
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/permetrics) 
[![PyPI version](https://badge.fury.io/py/permetrics.svg)](https://badge.fury.io/py/permetrics)
[![DOI](https://zenodo.org/badge/280617738.svg)](https://zenodo.org/badge/latestdoi/280617738)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Installation

Install the [current PyPI release](https://pypi.python.org/pypi/permetrics):

```bash
pip install permetrics
```

Or install the development version from GitHub:

```bash
pip install git+https://github.com/thieunguyen5991/permetrics
```


## Example
+ All you need to do is: (Make sure your y_true and y_pred is a numpy array)

```python 

## For example with RMSE:

from numpy import array
from permetrics.regression import Metrics

## 1-D array
y_true = array([3, -0.5, 2, 7])
y_pred = array([2.5, 0.0, 2, 8])

obj1 = Metrics(y_true, y_pred)
print(obj1.rmse_func(clean=True, decimal=5))

## > 1-D array
y_true = array([[0.5, 1], [-1, 1], [7, -6]])
y_pred = array([[0, 2], [-1, 2], [8, -5]])

multi_outputs = [None, "raw_values", [0.3, 1.2], array([0.5, 0.2]), (0.1, 0.9)]
obj2 = Metrics(y_true, y_pred)
for multi_output in multi_outputs:
    print(obj2.rmse_func(clean=False, multi_output=multi_output, decimal=5))

...
```

## References

#### Publications
+ If you see my code useful and use any a part of it, please cites my works here
```code 
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
```
 
+ This project related to my another projects which are "meta-heuristics" and "neural-network", check it here
    + https://github.com/thieunguyen5991/metaheuristics
    + https://github.com/chasebk

