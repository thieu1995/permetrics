# A framework of PERformance METRICS (PerMetrics) for artificial intelligence models
[![GitHub release](https://img.shields.io/badge/release-1.3.0-yellow.svg)]()
[![Documentation Status](https://readthedocs.org/projects/permetrics/badge/?version=latest)](https://permetrics.readthedocs.io/en/latest/?badge=latest)
[![](https://img.shields.io/badge/python-3.6+-orange.svg)](https://www.python.org/downloads/release/python-360/)
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/permetrics) 
[![PyPI version](https://badge.fury.io/py/permetrics.svg)](https://badge.fury.io/py/permetrics)
[![DOI](https://zenodo.org/badge/280617738.svg)](https://zenodo.org/badge/latestdoi/280617738)
[![Downloads](https://pepy.tech/badge/permetrics)](https://pepy.tech/project/permetrics)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


---
> "Knowledge is power, sharing it is the premise of progress in life. It seems like a burden to someone, but it is the only way to achieve immortality."
>  --- [Thieu Nguyen](https://www.researchgate.net/profile/Thieu_Nguyen6)
---

# Quick notification

* Add classification metrics to version 1.3.0
* Add more metrics to version 1.2.2
* The version 1.2.0 has serious problem with calculate multiple metrics (OOP style), please update to version 1.2.1 as 
  soon as possible for your sake.



## Introduction
* PerMetrics is a python library for performance metrics of machine learning models.

* The goals of this framework are:
    * Combine all metrics for regression, classification and clustering models
    * Helping users in all field access to metrics as fast as possible
    * Perform Qualitative Analysis of models.
    * Perform Quantitative Analysis of models.


### Dependencies
* Python (>= 3.6)
* Numpy (>= 1.15.1)


### User installation
Install the [current PyPI release](https://pypi.python.org/pypi/permetrics):

```bash
pip install permetrics==1.3.0
```

Or install the development version from GitHub:

```bash
pip install git+https://github.com/thieu1995/permetrics
```


### Example

+ Permetrics version >= 1.2.0

```python 

from numpy import array
from permetrics.regression import RegressionMetric

## For 1-D array
y_true = array([3, -0.5, 2, 7])
y_pred = array([2.5, 0.0, 2, 8])

evaluator = RegressionMetric(y_true, y_pred, decimal=5)
print(evaluator.RMSE())
print(evaluator.MSE())

## For > 1-D array
y_true = array([[0.5, 1], [-1, 1], [7, -6]])
y_pred = array([[0, 2], [-1, 2], [8, -5]])

evaluator = RegressionMetric(y_true, y_pred, decimal=5)
print(evaluator.RMSE(multi_output="raw_values", decimal=5))
print(evaluator.MAE(multi_output="raw_values", decimal=5))


## All metrics

EVS = evs = explained_variance_score
ME = me = max_error
MBE = mbe = mean_bias_error
MAE = mae = mean_absolute_error
MSE = mse = mean_squared_error
RMSE = rmse = root_mean_squared_error
MSLE = msle = mean_squared_log_error
MedAE = medae = median_absolute_error
MRE = mre = MRB = mrb = mean_relative_bias = mean_relative_error
MPE = mpe = mean_percentage_error
MAPE = mape = mean_absolute_percentage_error
SMAPE = smape = symmetric_mean_absolute_percentage_error
MAAPE = maape = mean_arctangent_absolute_percentage_error
MASE = mase = mean_absolute_scaled_error
NSE = nse = nash_sutcliffe_efficiency
NNSE = nnse = normalized_nash_sutcliffe_efficiency
WI = wi = willmott_index
R = r = PCC = pcc = pearson_correlation_coefficient
AR = ar = APCC = apcc = absolute_pearson_correlation_coefficient
R2s = r2s = pearson_correlation_coefficient_square
CI = ci = confidence_index
COD = cod = R2 = r2 = coefficient_of_determination
ACOD = acod = AR2 = ar2 = adjusted_coefficient_of_determination
DRV = drv = deviation_of_runoff_volume
KGE = kge = kling_gupta_efficiency
GINI = gini = gini_coefficient
GINI_WIKI = gini_wiki = gini_coefficient_wiki
PCD = pcd = prediction_of_change_in_direction
CE = ce = cross_entropy
KLD = kld = kullback_leibler_divergence
JSD = jsd = jensen_shannon_divergence
VAF = vaf = variance_accounted_for
RAE = rae = relative_absolute_error
A10 = a10 = a10_index
A20 = a20 = a20_index
A30 = a30 = a30_index
NRMSE = nrmse = normalized_root_mean_square_error
RSE = rse = residual_standard_error

RE = re = RB = rb = single_relative_bias = single_relative_error
AE = ae = single_absolute_error
SE = se = single_squared_error
SLE = sle = single_squared_log_error

```



+ Permetrics version <= 1.1.3

```python 
##  All you need to do is: (Make sure your y_true and y_pred is a numpy array)
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
    * https://github.com/thieu1995/mealpy
    * https://github.com/thieu1995/opfunu
    * https://github.com/thieu1995/metaheuristics
    * https://github.com/chasebk
    
### Metrics


| **Problem** | **STT** | **Metric ** | **Metric Fullname** | **Characteristics** |
|:---:|:---:|:---:|:---:|:---:|
| **Regression** | 1 | EVS | Explained Variance Score | Greater is better (Best = 1), Range=(-inf, 1.0] |
| **** | 2 | ME | Max Error | Smaller is better (Best = 0), Range=[0, +inf) |
| **** | 3 | MBE | Mean Bias Error | Best = 0, Range=(-inf, +inf) |
| **** | 4 | MAE | Mean Absolute Error | Smaller is better (Best = 0), Range=[0, +inf) |
| **** | 5 | MSE | Mean Squared Error | Smaller is better (Best = 0), Range=[0, +inf) |
| **** | 6 | RMSE | Root Mean Squared Error | Smaller is better (Best = 0), Range=[0, +inf) |
| **** | 7 | MSLE | Mean Squared Log Error | Smaller is better (Best = 0), Range=[0, +inf) |
| **** | 8 | MedAE | Median Absolute Error | Smaller is better (Best = 0), Range=[0, +inf) |
| **** | 9 | MRE / MRB | Mean Relative Error / Mean Relative Bias | Smaller is better (Best = 0), Range=[0, +inf) |
| **** | 10 | MPE | Mean Percentage Error | Best = 0, Range=(-inf, +inf) |
| **** | 11 | MAPE | Mean Absolute Percentage Error | Smaller is better (Best = 0), Range=[0, +inf) |
| **** | 12 | SMAPE | Symmetric Mean Absolute Percentage Error | Smaller is better (Best = 0), Range=[0, 1] |
| **** | 13 | MAAPE | Mean Arctangent Absolute Percentage Error | Smaller is better (Best = 0), Range=[0, +inf) |
| **** | 14 | MASE | Mean Absolute Scaled Error | Smaller is better (Best = 0), Range=[0, +inf) |
| **** | 15 | NSE | Nash-Sutcliffe Efficiency Coefficient | Greater is better (Best = 1), Range=(-inf, 1] |
| **** | 16 | NNSE | Normalized Nash-Sutcliffe Efficiency Coefficient | Greater is better (Best = 1), Range=[0, 1] |
| **** | 17 | WI | Willmott Index | Greater is better (Best = 1), Range=[0, 1] |
| **** | 18 | R / PCC | Pearson’s Correlation Coefficient | Greater is better (Best = 1), Range=[-1, 1] |
| **** | 19 | AR / APCC | Absolute Pearson's Correlation Coefficient | Greater is better (Best = 1), Range=[-1, 1] |
| **** | 20 | R2s | (Pearson’s Correlation Index) ^ 2 | Greater is better (Best = 1), Range=[0, 1] |
| **** | 21 | R2 / COD | Coefficient of Determination | Greater is better (Best = 1), Range=(-inf, 1] |
| **** | 22 | AR2 / ACOD | Adjusted Coefficient of Determination | Greater is better (Best = 1), Range=(-inf, 1] |
| **** | 23 | CI | Confidence Index | Greater is better (Best = 1), Range=(-inf, 1] |
| **** | 24 | DRV |  Deviation of Runoff Volume | Smaller is better (Best = 1.0), Range=[1, +inf) |
| **** | 25 | KGE | Kling-Gupta Efficiency | Greater is better (Best = 1), Range=(-inf, 1] |
| **** | 26 | GINI | Gini Coefficient | Smaller is better (Best = 0), Range=[0, +inf) |
| **** | 27 | GINI_WIKI | Gini Coefficient on Wikipage | Smaller is better (Best = 0), Range=[0, +inf) |
| **** | 28 | PCD | Prediction of Change in Direction | Greater is better (Best = 1.0), Range=[0, 1] |
| **** | 29 | CE | Cross Entropy | Range(-inf, 0], Can't give comment about this |
| **** | 30 | KLD | Kullback Leibler Divergence | Best = 0, Range=(-inf, +inf) |
| **** | 31 | JSD | Jensen Shannon Divergence | Smaller is better (Best = 0), Range=[0, +inf) |
| **** | 32 | VAF | Variance Accounted For | Greater is better (Best = 100%), Range=(-inf, 100%] |
| **** | 33 | RAE | Relative Absolute Error | Smaller is better (Best = 0), Range=[0, +inf) |
| **** | 34 | A10 | A10 Index | Greater is better (Best = 1), Range=[0, 1] |
| **** | 35 | A20 | A20 Index | Greater is better (Best = 1), Range=[0, 1] |
| **** | 36 | A30 | A30 Index | Greater is better (Best = 1), Range=[0, 1] |
| **** | 37 | NRMSE | Normalized Root Mean Square Error | Smaller is better (Best = 0), Range=[0, +inf) |
| **** | 38 | RSE | Residual Standard Error | Smaller is better (Best = 0), Range=[0, +inf) |
| **** | 39 | RE / RB | Relative Error / Relative Bias | Best = 0, Range=(-inf, +inf) |
| **** | 40 | AE | Absolute Error | Best = 0, Range=(-inf, +inf) |
| **** | 41 | SE |  Squared Error | Smaller is better (Best = 0), Range=[0, +inf) |
| **** | 42 | SLE | Squared Log Error | Smaller is better (Best = 0), Range=[0, +inf) |
| **** | 43 |  |  |  |
| **Classification** | 1 | PS | Precision Score | Higher is better (Best = 1), Range = [0, 1] |
| **** | 2 | NPV | Negative Predictive Value | Higher is better (Best = 1), Range = [0, 1] |
| **** | 3 | RS | Recall Score | Higher is better (Best = 1), Range = [0, 1] |
| **** | 4 | AS | Accuracy Score | Higher is better (Best = 1), Range = [0, 1] |
| **** | 5 | F1S | F1 Score | Higher is better (Best = 1), Range = [0, 1] |
| **** | 6 | F2S | F2 Score | Higher is better (Best = 1), Range = [0, 1] |
| **** | 7 | FBS | F-Beta Score | Higher is better (Best = 1), Range = [0, 1] |
| **** | 8 | SS | Specificity Score | Higher is better (Best = 1), Range = [0, 1] |
| **** | 9 | MCC | Matthews Correlation Coefficient | Higher is better (Best = 1), Range = [-1, +1] |
| **** | 10 | HL | Hamming Loss | Higher is better (Best = 1), Range = [0, 1] |
| **** | 11 | LS | Lift Score | Higher is better (Best = 0), Range = [0, +inf) |
| **** | 12 |






| **Problem** | **STT** | **Metric** | **Metric Fullname** | **Characteristics** |
|---|---|---|---|---|
| Regression | 1 | EVS | Explained Variance Score | Greater is better (Best = 1), Range=(-inf, 1.0] |
|  | 2 | ME | Max Error | Smaller is better (Best = 0), Range=[0, +inf) |
|  | 3 | MBE | Mean Bias Error | Best = 0, Range=(-inf, +inf) |
|  | 4 | MAE | Mean Absolute Error | Smaller is better (Best = 0), Range=[0, +inf) |
|  | 5 | MSE | Mean Squared Error | Smaller is better (Best = 0), Range=[0, +inf) |
|  | 6 | RMSE | Root Mean Squared Error | Smaller is better (Best = 0), Range=[0, +inf) |
|  | 7 | MSLE | Mean Squared Log Error | Smaller is better (Best = 0), Range=[0, +inf) |
|  | 8 | MedAE | Median Absolute Error | Smaller is better (Best = 0), Range=[0, +inf) |
|  | 9 | MRE / MRB | Mean Relative Error / Mean Relative Bias | Smaller is better (Best = 0), Range=[0, +inf) |
|  | 10 | MPE | Mean Percentage Error | Best = 0, Range=(-inf, +inf) |
|  | 11 | MAPE | Mean Absolute Percentage Error | Smaller is better (Best = 0), Range=[0, +inf) |
|  | 12 | SMAPE | Symmetric Mean Absolute Percentage Error | Smaller is better (Best = 0), Range=[0, 1] |
|  | 13 | MAAPE | Mean Arctangent Absolute Percentage Error | Smaller is better (Best = 0), Range=[0, +inf) |
|  | 14 | MASE | Mean Absolute Scaled Error | Smaller is better (Best = 0), Range=[0, +inf) |
|  | 15 | NSE | Nash-Sutcliffe Efficiency Coefficient | Greater is better (Best = 1), Range=(-inf, 1] |
|  | 16 | NNSE | Normalized Nash-Sutcliffe Efficiency Coefficient | Greater is better (Best = 1), Range=[0, 1] |
|  | 17 | WI | Willmott Index | Greater is better (Best = 1), Range=[0, 1] |
|  | 18 | R / PCC | Pearson’s Correlation Coefficient | Greater is better (Best = 1), Range=[-1, 1] |
|  | 19 | AR / APCC | Absolute Pearson's Correlation Coefficient | Greater is better (Best = 1), Range=[-1, 1] |
|  | 20 | R2s | (Pearson’s Correlation Index) ^ 2 | Greater is better (Best = 1), Range=[0, 1] |
|  | 21 | R2 / COD | Coefficient of Determination | Greater is better (Best = 1), Range=(-inf, 1] |
|  | 22 | AR2 / ACOD | Adjusted Coefficient of Determination | Greater is better (Best = 1), Range=(-inf, 1] |
|  | 23 | CI | Confidence Index | Greater is better (Best = 1), Range=(-inf, 1] |
|  | 24 | DRV |  Deviation of Runoff Volume | Smaller is better (Best = 1.0), Range=[1, +inf) |
|  | 25 | KGE | Kling-Gupta Efficiency | Greater is better (Best = 1), Range=(-inf, 1] |
|  | 26 | GINI | Gini Coefficient | Smaller is better (Best = 0), Range=[0, +inf) |
|  | 27 | GINI_WIKI | Gini Coefficient on Wikipage | Smaller is better (Best = 0), Range=[0, +inf) |
|  | 28 | PCD | Prediction of Change in Direction | Greater is better (Best = 1.0), Range=[0, 1] |
|  | 29 | CE | Cross Entropy | Range(-inf, 0], Can't give comment about this |
|  | 30 | KLD | Kullback Leibler Divergence | Best = 0, Range=(-inf, +inf) |
|  | 31 | JSD | Jensen Shannon Divergence | Smaller is better (Best = 0), Range=[0, +inf) |
|  | 32 | VAF | Variance Accounted For | Greater is better (Best = 100%), Range=(-inf, 100%] |
|  | 33 | RAE | Relative Absolute Error | Smaller is better (Best = 0), Range=[0, +inf) |
|  | 34 | A10 | A10 Index | Greater is better (Best = 1), Range=[0, 1] |
|  | 35 | A20 | A20 Index | Greater is better (Best = 1), Range=[0, 1] |
|  | 36 | A30 | A30 Index | Greater is better (Best = 1), Range=[0, 1] |
|  | 37 | NRMSE | Normalized Root Mean Square Error | Smaller is better (Best = 0), Range=[0, +inf) |
|  | 38 | RSE | Residual Standard Error | Smaller is better (Best = 0), Range=[0, +inf) |
|  | 39 | RE / RB | Relative Error / Relative Bias | Best = 0, Range=(-inf, +inf) |
|  | 40 | AE | Absolute Error | Best = 0, Range=(-inf, +inf) |
|  | 41 | SE |  Squared Error | Smaller is better (Best = 0), Range=[0, +inf) |
|  | 42 | SLE | Squared Log Error | Smaller is better (Best = 0), Range=[0, +inf) |
|  | 43 |  |  |  |
| Classification | 1 | MLL | Mean Log Likelihood |  |
|  | 2 | LL | Log Likelihood |


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
+ Multiclass log loss
+ cross-entropy
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
+ Discrete Rater Comparison (confusion matrix)
 
