
<p align="center">
<img style="max-width:100%;" 
src="https://thieu1995.github.io/post/2023-08/permetrics-01.png" 
alt="PERMETRICS"/>
</p>


---

[![GitHub release](https://img.shields.io/badge/release-1.4.3-yellow.svg)](https://github.com/thieu1995/permetrics/releases)
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/permetrics) 
[![PyPI version](https://badge.fury.io/py/permetrics.svg)](https://badge.fury.io/py/permetrics)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/permetrics.svg)
![PyPI - Status](https://img.shields.io/pypi/status/permetrics.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/permetrics.svg)
[![Downloads](https://pepy.tech/badge/permetrics)](https://pepy.tech/project/permetrics)
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
field access metrics as fast as possible

* **Free software:** GNU General Public License (GPL) V3 license
* **Total metrics**: 107 (47 regression metrics, 20 classification metrics, 40 clustering metrics)
* **Documentation:** https://permetrics.readthedocs.io/en/latest/
* **Python versions:** >= 3.7.x
* **Dependencies:** numpy, scipy


# Notification

* **Currently, there is a huge misunderstanding among frameworks around the world about the notation of R, R2, and R^2.** 
* Please read the file [R-R2-Rsquared.docx](https://github.com/thieu1995/permetrics/blob/master/R-R2-Rsquared.docx) to understand the differences between them and why there is such confusion.

<p align="center"><img src=".github/assets/rr2.png" alt="R" title="R"/></p>

* **My recommendation is to denote the Coefficient of Determination as COD or R2, while the squared Pearson's 
  Correlation Coefficient should be denoted as R^2 or RSQ (as in Excel software).**


# Installation

### Install with pip
Install the [current PyPI release](https://pypi.python.org/pypi/permetrics):
```sh 
$ pip install permetrics==1.4.3
```

Or installing from the source code, use:
```sh 
$ git clone https://github.com/thieu1995/permetrics.git
$ cd permetrics
$ python setup.py install
```

Or install the development version from GitHub:
```bash
pip install git+https://github.com/thieu1995/permetrics
```

After installation, you can import Permetrics as any other Python module:

```sh
$ python
>>> import permetrics
>>> permetrics.__version__
```

**Let's go through some examples. The more complicated test case in the folder: examples**

The [documentation](https://permetrics.readthedocs.io/) includes more detailed installation instructions and explanations.


### Example with Regression metrics


```python
import numpy as np
from permetrics import RegressionMetric

## For 1-D array
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

evaluator = RegressionMetric(y_true, y_pred, decimal=5)
print(evaluator.RMSE())
print(evaluator.MSE())

## For > 1-D array
y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
y_pred = np.array([[0, 2], [-1, 2], [8, -5]])

evaluator = RegressionMetric(y_true, y_pred, decimal=5)
print(evaluator.RMSE(multi_output="raw_values", decimal=5))
print(evaluator.MAE(multi_output="raw_values", decimal=5))
```


### Example with Classification metrics

```python
from permetrics import ClassificationMetric

## For integer labels or categorical labels
y_true = [0, 1, 0, 0, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1]

# y_true = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]

evaluator = ClassificationMetric(y_true, y_pred, decimal=5)

## Call specific function inside object, each function has 3 names like below

print(evaluator.f1_score())
print(evaluator.F1S(average="micro"))
print(evaluator.F1S(average="macro"))
print(evaluator.F1S(average="weighted"))

```

### Example with Clustering metrics

```python
import numpy as np
from permetrics import ClusteringMetric

# generate sample data
X = np.random.uniform(-1, 10, size=(500, 7))        # 500 examples, 7 features
y_true = np.random.randint(0, 4, size=500)          # 4 clusters
y_pred = np.random.randint(0, 4, size=500)

evaluator = ClusteringMetric(y_true=y_true, y_pred=y_pred, X=X, decimal=5)

## Call specific function inside object, each function has 2 names (fullname and short name)
##    + Internal metrics: Need X and y_pred and has suffix as index
##    + External metrics: Need y_true and y_pred and has suffix as score

print(evaluator.ball_hall_index())
print(evaluator.BHI())
```

### Metrics

<table><thead><tr><th>Problem</th><th>STT</th><th>Metric</th><th>Metric Fullname</th><th>Characteristics</th></tr></thead><tbody><tr><th>Regression</th><td>1</td><td>EVS</td><td>Explained Variance Score</td><td>Bigger is better (Best = 1), Range=(-inf, 1.0]</td></tr><tr><th>****</th><td>2</td><td>ME</td><td>Max Error</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>3</td><td>MBE</td><td>Mean Bias Error</td><td>Best = 0, Range=(-inf, +inf)</td></tr><tr><th>****</th><td>4</td><td>MAE</td><td>Mean Absolute Error</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>5</td><td>MSE</td><td>Mean Squared Error</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>6</td><td>RMSE</td><td>Root Mean Squared Error</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>7</td><td>MSLE</td><td>Mean Squared Log Error</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>8</td><td>MedAE</td><td>Median Absolute Error</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>9</td><td>MRE / MRB</td><td>Mean Relative Error / Mean Relative Bias</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>10</td><td>MPE</td><td>Mean Percentage Error</td><td>Best = 0, Range=(-inf, +inf)</td></tr><tr><th>****</th><td>11</td><td>MAPE</td><td>Mean Absolute Percentage Error</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>12</td><td>SMAPE</td><td>Symmetric Mean Absolute Percentage Error</td><td>Smaller is better (Best = 0), Range=[0, 1]</td></tr><tr><th>****</th><td>13</td><td>MAAPE</td><td>Mean Arctangent Absolute Percentage Error</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>14</td><td>MASE</td><td>Mean Absolute Scaled Error</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>15</td><td>NSE</td><td>Nash-Sutcliffe Efficiency Coefficient</td><td>Bigger is better (Best = 1), Range=(-inf, 1]</td></tr><tr><th>****</th><td>16</td><td>NNSE</td><td>Normalized Nash-Sutcliffe Efficiency Coefficient</td><td>Bigger is better (Best = 1), Range=[0, 1]</td></tr><tr><th>****</th><td>17</td><td>WI</td><td>Willmott Index</td><td>Bigger is better (Best = 1), Range=[0, 1]</td></tr><tr><th>****</th><td>18</td><td>R / PCC</td><td>Pearson’s Correlation Coefficient</td><td>Bigger is better (Best = 1), Range=[-1, 1]</td></tr><tr><th>****</th><td>19</td><td>AR / APCC</td><td>Absolute Pearson&#39;s Correlation Coefficient</td><td>Bigger is better (Best = 1), Range=[-1, 1]</td></tr><tr><th>****</th><td>20</td><td>RSQ/R2S</td><td>(Pearson’s Correlation Index) ^ 2</td><td>Bigger is better (Best = 1), Range=[0, 1]</td></tr><tr><th>****</th><td>21</td><td>R2 / COD</td><td>Coefficient of Determination</td><td>Bigger is better (Best = 1), Range=(-inf, 1]</td></tr><tr><th>****</th><td>22</td><td>AR2 / ACOD</td><td>Adjusted Coefficient of Determination</td><td>Bigger is better (Best = 1), Range=(-inf, 1]</td></tr><tr><th>****</th><td>23</td><td>CI</td><td>Confidence Index</td><td>Bigger is better (Best = 1), Range=(-inf, 1]</td></tr><tr><th>****</th><td>24</td><td>DRV</td><td>Deviation of Runoff Volume</td><td>Smaller is better (Best = 1.0), Range=[1, +inf)</td></tr><tr><th>****</th><td>25</td><td>KGE</td><td>Kling-Gupta Efficiency</td><td>Bigger is better (Best = 1), Range=(-inf, 1]</td></tr><tr><th>****</th><td>26</td><td>GINI</td><td>Gini Coefficient</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>27</td><td>GINI_WIKI</td><td>Gini Coefficient on Wikipage</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>28</td><td>PCD</td><td>Prediction of Change in Direction</td><td>Bigger is better (Best = 1.0), Range=[0, 1]</td></tr><tr><th>****</th><td>29</td><td>CE</td><td>Cross Entropy</td><td>Range(-inf, 0], Can&#39;t give comment about this</td></tr><tr><th>****</th><td>30</td><td>KLD</td><td>Kullback Leibler Divergence</td><td>Best = 0, Range=(-inf, +inf)</td></tr><tr><th>****</th><td>31</td><td>JSD</td><td>Jensen Shannon Divergence</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>32</td><td>VAF</td><td>Variance Accounted For</td><td>Bigger is better (Best = 100%), Range=(-inf, 100%]</td></tr><tr><th>****</th><td>33</td><td>RAE</td><td>Relative Absolute Error</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>34</td><td>A10</td><td>A10 Index</td><td>Bigger is better (Best = 1), Range=[0, 1]</td></tr><tr><th>****</th><td>35</td><td>A20</td><td>A20 Index</td><td>Bigger is better (Best = 1), Range=[0, 1]</td></tr><tr><th>****</th><td>36</td><td>A30</td><td>A30 Index</td><td>Bigger is better (Best = 1), Range=[0, 1]</td></tr><tr><th>****</th><td>37</td><td>NRMSE</td><td>Normalized Root Mean Square Error</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>38</td><td>RSE</td><td>Residual Standard Error</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>39</td><td>RE / RB</td><td>Relative Error / Relative Bias</td><td>Best = 0, Range=(-inf, +inf)</td></tr><tr><th>****</th><td>40</td><td>AE</td><td>Absolute Error</td><td>Best = 0, Range=(-inf, +inf)</td></tr><tr><th>****</th><td>41</td><td>SE</td><td>Squared Error</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>42</td><td>SLE</td><td>Squared Log Error</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>43</td><td>COV</td><td>Covariance</td><td>Bigger is better (No best value), Range=(-inf, +inf)</td></tr><tr><th>****</th><td>44</td><td>COR</td><td>Correlation</td><td>Bigger is better (Best = 1), Range=[-1, +1]</td></tr><tr><th>****</th><td>45</td><td>EC</td><td>Efficiency Coefficient</td><td>Bigger is better (Best = 1), Range=(-inf, +1]</td></tr><tr><th>****</th><td>46</td><td>OI</td><td>Overall Index</td><td>Bigger is better (Best = 1), Range=(-inf, +1]</td></tr><tr><th>****</th><td>47</td><td>CRM</td><td>Coefficient of Residual Mass</td><td>Smaller is better (Best = 0), Range=(-inf, +inf)</td></tr><tr><th>****</th><td>****</td><td>****</td><td>****</td><td>****</td></tr><tr><th>Classification</th><td>1</td><td>PS</td><td>Precision Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>2</td><td>NPV</td><td>Negative Predictive Value</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>3</td><td>RS</td><td>Recall Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>4</td><td>AS</td><td>Accuracy Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>5</td><td>F1S</td><td>F1 Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>6</td><td>F2S</td><td>F2 Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>7</td><td>FBS</td><td>F-Beta Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>8</td><td>SS</td><td>Specificity Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>9</td><td>MCC</td><td>Matthews Correlation Coefficient</td><td>Bigger is better (Best = 1), Range = [-1, +1]</td></tr><tr><th>****</th><td>10</td><td>HS</td><td>Hamming Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>11</td><td>CKS</td><td>Cohen&#39;s kappa score</td><td>Bigger is better (Best = +1), Range = [-1, +1]</td></tr><tr><th>****</th><td>12</td><td>JSI</td><td>Jaccard Similarity Coefficient</td><td>Bigger is better (Best = +1), Range = [0, +1]</td></tr><tr><th>****</th><td>13</td><td>GMS</td><td>Geometric Mean Score</td><td>Bigger is better (Best = +1), Range = [0, +1]</td></tr><tr><th>****</th><td>14</td><td>ROC-AUC</td><td>ROC-AUC</td><td>Bigger is better (Best = +1), Range = [0, +1]</td></tr><tr><th>****</th><td>15</td><td>LS</td><td>Lift Score</td><td>Bigger is better (No best value), Range = [0, +inf)</td></tr><tr><th>****</th><td>16</td><td>GINI</td><td>GINI Index</td><td>Smaller is better (Best = 0), Range = [0, +1]</td></tr><tr><th>****</th><td>17</td><td>CEL</td><td>Cross Entropy Loss</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>18</td><td>HL</td><td>Hinge Loss</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>19</td><td>KLDL</td><td>Kullback Leibler Divergence Loss</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>20</td><td>BSL</td><td>Brier Score Loss</td><td>Smaller is better (Best = 0), Range=[0, +1]</td></tr><tr><th>****</th><td>****</td><td>****</td><td>****</td><td>****</td></tr><tr><th>Clustering</th><td>1</td><td>BHI</td><td>Ball Hall Index</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>2</td><td>XBI</td><td>Xie Beni Index</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>3</td><td>DBI</td><td>Davies Bouldin Index</td><td>Smaller is better (Best = 0), Range=[0, +inf)</td></tr><tr><th>****</th><td>4</td><td>BRI</td><td>Banfeld Raftery Index</td><td>Smaller is better (No best value), Range=(-inf, inf)</td></tr><tr><th>****</th><td>5</td><td>KDI</td><td>Ksq Detw Index</td><td>Smaller is better (No best value), Range=(-inf, +inf)</td></tr><tr><th>****</th><td>6</td><td>DRI</td><td>Det Ratio Index</td><td>Bigger is better (No best value), Range=[0, +inf)</td></tr><tr><th>****</th><td>7</td><td>DI</td><td>Dunn Index</td><td>Bigger is better (No best value), Range=[0, +inf)</td></tr><tr><th>****</th><td>8</td><td>CHI</td><td>Calinski Harabasz Index</td><td>Bigger is better (No best value), Range=[0, inf)</td></tr><tr><th>****</th><td>9</td><td>LDRI</td><td>Log Det Ratio Index</td><td>Bigger is better (No best value), Range=(-inf, +inf)</td></tr><tr><th>****</th><td>10</td><td>LSRI</td><td>Log SS Ratio Index</td><td>Bigger is better (No best value), Range=(-inf, +inf)</td></tr><tr><th>****</th><td>11</td><td>SI</td><td>Silhouette Index</td><td>Bigger is better (Best = 1), Range = [-1, +1]</td></tr><tr><th>****</th><td>12</td><td>SSEI</td><td>Sum of Squared Error Index</td><td>Smaller is better (Best = 0), Range = [0, +inf)</td></tr><tr><th>****</th><td>13</td><td>DHI</td><td>Duda-Hart Index</td><td>Smaller is better (Best = 0), Range = [0, +inf)</td></tr><tr><th>****</th><td>14</td><td>BI</td><td>Beale Index</td><td>Smaller is better (Best = 0), Range = [0, +inf)</td></tr><tr><th>****</th><td>15</td><td>RSI</td><td>R-squared Index</td><td>Bigger is better (Best=1), Range = (-inf, 1]</td></tr><tr><th>****</th><td>16</td><td>DBCVI</td><td>Density-based Clustering Validation Index</td><td>Bigger is better (Best=0), Range = [0, 1]</td></tr><tr><th>****</th><td>17</td><td>HI</td><td>Hartigan Index</td><td>Bigger is better (best=0), Range = [0, +inf)</td></tr><tr><th>****</th><td>18</td><td>MIS</td><td>Mutual Info Score</td><td>Bigger is better (No best value), Range = [0, +inf)</td></tr><tr><th>****</th><td>19</td><td>NMIS</td><td>Normalized Mutual Info Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>20</td><td>RaS</td><td>Rand Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>21</td><td>FMS</td><td>Fowlkes Mallows Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>22</td><td>HS</td><td>Homogeneity Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>23</td><td>CS</td><td>Completeness Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>24</td><td>VMS</td><td>V-Measure Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>25</td><td>PrS</td><td>Precision Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>26</td><td>ReS</td><td>Recall Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>27</td><td>FmS</td><td>F-Measure Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>28</td><td>CDS</td><td>Czekanowski Dice Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>29</td><td>HGS</td><td>Hubert Gamma Score</td><td>Bigger is better (Best = 1), Range=[-1, +1]</td></tr><tr><th>****</th><td>30</td><td>JS</td><td>Jaccard Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>31</td><td>KS</td><td>Kulczynski Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>32</td><td>MNS</td><td>Mc Nemar Score</td><td>Bigger is better (No best value), Range=(-inf, +inf)</td></tr><tr><th>****</th><td>33</td><td>PhS</td><td>Phi Score</td><td>Bigger is better (No best value), Range = (-inf, +inf)</td></tr><tr><th>****</th><td>34</td><td>RTS</td><td>Rogers Tanimoto Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>35</td><td>RRS</td><td>Russel Rao Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>36</td><td>SS1S</td><td>Sokal Sneath1 Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>37</td><td>SS2S</td><td>Sokal Sneath2 Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>38</td><td>PuS</td><td>Purity Score</td><td>Bigger is better (Best = 1), Range = [0, 1]</td></tr><tr><th>****</th><td>39</td><td>ES</td><td>Entropy Score</td><td>Smaller is better (Best = 0), Range = [0, 1]</td></tr><tr><th>****</th><td>40</td><td>TS</td><td>Tau Score</td><td>Bigger is better (Best = 1), Range = [-1, 1]</td></tr><tr><th>****</th><td>****</td><td>****</td><td>****</td><td>****</td></tr></tbody></table>


# Support (questions, problems)


### Official channels 

* Official source code repo: https://github.com/thieu1995/permetrics
* Official document: https://permetrics.readthedocs.io/
* Download releases: https://pypi.org/project/permetrics/
* Issue tracker: https://github.com/thieu1995/permetrics/issues
* Notable changes log: https://github.com/thieu1995/permetrics/blob/master/ChangeLog.md
* Official chat group: https://t.me/+fRVCJGuGJg1mNDg1

* This project also related to our another projects which are "optimization" and "machine learning", check it here:
    * https://github.com/thieu1995/mealpy
    * https://github.com/thieu1995/metaheuristics
    * https://github.com/thieu1995/opfunu
    * https://github.com/thieu1995/enoppy
    * https://github.com/thieu1995/MetaCluster
    * https://github.com/thieu1995/pfevaluator
    * https://github.com/thieu1995/mafese
    * https://github.com/aiir-team


### Citation Request 

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

@article{van2023mealpy,
  title={MEALPY: An open-source library for latest meta-heuristic algorithms in Python},
  author={Van Thieu, Nguyen and Mirjalili, Seyedali},
  journal={Journal of Systems Architecture},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.sysarc.2023.102871}
}
```

### Related Documents

1) https://www.debadityachakravorty.com/ai-ml/cmatrix/
2) https://neptune.ai/blog/evaluation-metrics-binary-classification
3) https://danielyang1009.github.io/model-performance-measure/
4) https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
5) http://cran.nexr.com/web/packages/clusterCrit/vignettes/clusterCrit.pdf
6) https://torchmetrics.readthedocs.io/en/latest/
7) http://rasbt.github.io/mlxtend/user_guide/evaluate/lift_score/
8) https://www.baeldung.com/cs/multi-class-f1-score
9) https://kavita-ganesan.com/how-to-compute-precision-and-recall-for-a-multi-class-classification-problem/#.YoXMSqhBy3A
10) https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/
