# A framework of PERformance METRICS (PerMetrics) for artificial intelligence models
[![GitHub release](https://img.shields.io/badge/release-1.1.3-yellow.svg)]()
[![Documentation Status](https://readthedocs.org/projects/permetrics/badge/?version=latest)](https://permetrics.readthedocs.io/en/latest/?badge=latest)
[![](https://img.shields.io/badge/python-3.7+-orange.svg)](https://www.python.org/downloads/release/python-370/)
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/permetrics) 
[![PyPI version](https://badge.fury.io/py/permetrics.svg)](https://badge.fury.io/py/permetrics)
[![DOI](https://zenodo.org/badge/280617738.svg)](https://zenodo.org/badge/latestdoi/280617738)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


---
> "Knowledge is power, sharing it is the premise of progress in life. It seems like a burden to someone, but it is the only way to achieve immortality."
>  --- [Thieu Nguyen](https://www.researchgate.net/profile/Thieu_Nguyen6)
---

## Introduction
* PerMetrics is a python library for performance metrics of machine learning models.

* The goals of this framework are:
    * Combine all metrics for regression, classification and clustering models
    * Helping users in all field access to metrics as fast as possible
    * Perform Qualitative Analysis of models.
    * Perform Quantitative Analysis of models.

* Metrics

| Problem            | STT    |  Metric    | Metric Fullname                            | Characteristics                |
|----------------	|:---:	|:---------:	|-------------------------------------------	|------------------------------	|
| Regression        |  1    |    EVS        | Explained Variance Score                    | Larger is better (Best = 1)    |
|                	|  2    |     ME        | Max Error                                    | Smaller is better (Best = 0)    |
|                	|  3    |    MAE        | Mean Absolute Error                        | Smaller is better (Best = 0)    |
|                	|  4    |    MSE        | Mean Squared Error                            | Smaller is better (Best = 0)    |
|                	|  5    |    RMSE    | Root Mean Squared Error                    | Smaller is better (Best = 0)    |
|                	|  6    |    MSLE    | Mean Squared Log Error                        | Smaller is better (Best = 0)    |
|                	|  7    |   MedAE    | Median Absolute Error                        | Smaller is better (Best = 0)    |
|                	|  8    |    MRE        | Mean Relative Error                        | Smaller is better (Best = 0)    |
|                	|  9    |    MAPE    | Mean Absolute Percentage Error                | Smaller is better (Best = 0)    |
|                	|  10    |   SMAPE    | Symmetric Mean Absolute Percentage Error    | Smaller is better (Best = 0)    |
|                	|  11    |   MAAPE    | Mean Arctangent Absolute Percentage Error    | Smaller is better (Best = 0)    |
|                	|  12    |    MASE    | Mean Absolute Scaled Error                    | Smaller is better (Best = 0)    |
|                	|  13    |    NSE        | Nash-Sutcliffe Efficiency Coefficient        | Larger is better (Best = 1)    |
|                	|  14    |     WI        | Willmott Index                                | Larger is better (Best = 1)    |
|                	|  15    |     R        | Pearson’s Correlation Index                | Larger is better (Best = 1)    |
|                	|  16    |     CI        | Confidence Index                            | Larger is better (Best = 1)    |
|                	|  17    |     R2        | Coefficient of Determination                | Larger is better (Best = 1)    |
|                	|  18    |    R2s        | (Pearson’s Correlation Index) ^ 2            | Larger is better (Best = 1)    |
|                	|  19    |    DRV        |  Deviation of Runoff   Volume                | Smaller is better (Best = 0)    |
|                	|  20    |    KGE        | Kling-Gupta Efficiency                        | Larger is better (Best = 1)    |
|                	|  21    |    GINI    | Gini Coefficient                            |                              	|
|                	|  22    | GINI_WIKI    | Gini Coefficient in Wiki                    |                              	|
|                	|  23    |    PCD        | Prediction of Change in Direction            |                              	|
|                	|  24    |     E        | Entropy                                    |                              	|
|                	|  25    |     CE        | Cross Entropy                                |                              	|
|                	|  26    |    KLD        | Kullback Leibler Divergence                |                              	|
|                	|  27    |    JSD        | Jensen Shannon Divergence                    |                              	|
|                	|  28    |    VAF        | Variance Accounted For                        |                              	|
|                	|  29    |    RAE        | Relative Absolute Error                    |                              	|
|                	|  30    |    A10        | A10 Index                                    |                              	|
|                	|  31    |    A20        | A20 Index                                    |                              	|
|                	|  32    |   NRMSE    | Normalized Root Mean Square Error            |                              	|
|                	|  33    |         RSE |        Residual Standard Error            	|                              	|
| Single Loss        |  1    |     RE        | Relative error                                | Smaller is better (Best = 0)    |
|                	|  2    |     AE        | Absolute error                                | Smaller is better (Best = 0)    |
|                	|  3    |     SE        |  Squared error                                | Smaller is better (Best = 0)    |
|                	|  4    |    SLE        | Squared log error                            | Smaller is better (Best = 0)    |
|                	|  5    |     LL        | Log likelihood                                | Smaller is better (Best = 0)    |
|                	|  6    |           	|                                           	|                              	|
| Classification    |  1    |    MLL        | Mean Log Likelihood                        | Smaller is better (Best = 0)    |
|                	|  2    |           	|                                           	|                              	|
| Clustering        |  1    |           	|                                           	|                              	|
|                	|  2    |           	|                                           	|                              	|

### Dependencies
* Python (>= 3.6)
* Numpy (>= 1.15.1)


### User installation
Install the [current PyPI release](https://pypi.python.org/pypi/permetrics):

```bash
pip install permetrics
```

Or install the development version from GitHub:

```bash
pip install git+https://github.com/thieu1995/permetrics
```


### Example
+ All you need to do is: (Make sure your y_true and y_pred is a numpy array)

```python 
#Simple example:

## For example with RMSE:

from numpy import array
from permetrics.regression import Metrics

## 1-D array
y_true = array([3, -0.5, 2, 7])
y_pred = array([2.5, 0.0, 2, 8])

y_true2 = array([3, -0.5, 2, 7])
y_pred2 = array([2.5, 0.0, 2, 9])

### C1. Using OOP style - very powerful when calculating multiple metrics
obj1 = Metrics(y_true, y_pred)  # Pass the data here
result = obj1.root_mean_squared_error(clean=True, decimal=5)
print(f"1-D array, OOP style: {result}")

### C2. Using functional style
obj2 = Metrics()
result = obj2.root_mean_squared_error(clean=True, decimal=5, y_true=y_true2, y_pred=y_pred2)  
# Pass the data here, remember the keywords (y_true, y_pred)
print(f"1-D array, Functional style: {result}")

## > 1-D array - Multi-dimensional Array
y_true = array([[0.5, 1], [-1, 1], [7, -6]])
y_pred = array([[0, 2], [-1, 2], [8, -5]])

multi_outputs = [None, "raw_values", [0.3, 1.2], array([0.5, 0.2]), (0.1, 0.9)]
obj3 = Metrics(y_true, y_pred)
for multi_output in multi_outputs:
    result = obj3.root_mean_squared_error(clean=False, multi_output=multi_output, decimal=5)
    print(f"n-D array, OOP style: {result}")

# Or run the simple:
python examples/RMSE.py

# The more complicated tests in the folder: examples
```
The [documentation](https://permetrics.readthedocs.io/) includes more detailed installation instructions and explanations.

### Changelog
* See the [ChangeLog.md](https://github.com/thieu1995/permetrics/blob/master/ChangeLog.md) for a history of notable changes to permetrics.


### Important links

* Official source code repo: https://github.com/thieu1995/permetrics
* Official documentation: https://permetrics.readthedocs.io/
* Download releases: https://pypi.org/project/permetrics/
* Issue tracker: https://github.com/thieu1995/permetrics/issues

* This project also related to my another projects which are "meta-heuristics" and "neural-network", check it here
    * https://github.com/thieu1995/opfunu
    * https://github.com/thieu1995/metaheuristics
    * https://github.com/chasebk
    
   
## Contributions 

### Citation
+ If you use permetrics in your project, please cite my works: 
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

# Future works
### Classification 
+ F1 score
+ Multiclass log loss
+ Lift
+ Average Precision for binary classification
+ precision / recall break-even point
+ cross-entropy
+ True Pos / False Pos / True Neg / False Neg rates
+ precision / recall / sensitivity / specificity
+ mutual information

### HIGHER LEVEL TRANSFORMATIONS TO HANDLE
+ GroupBy / Reduce
+ Weight individual samples or groups


### PROPERTIES METRICS CAN HAVE
+ Min or Max (optimize through minimization or maximization)
+ Binary Classification
    + Scores predicted class labels
    + Scores predicted ranking (most likely to least likely for being in one class)
    + Scores predicted probabilities
+ Multiclass Classification
    + Scores predicted class labels
    + Scores predicted probabilities
+ Regression (More)
+ Discrete Rater Comparison (confusion matrix)
 
