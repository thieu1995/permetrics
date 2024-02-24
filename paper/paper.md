---
title: 'PerMetrics: A Framework of Performance Metrics for Machine Learning Models'
tags:
  - model assessment tools
  - performance metrics
  - classification validation metrics
  - regression evaluation criteria
  - clustering criterion indices
  - machine learning metrics
authors:
  - name: Nguyen Van Thieu
    orcid: 0000-0001-9994-8747
    affiliation: 1
affiliations:
  - name: Faculty of Computer Science, Phenikaa University, Yen Nghia, Ha Dong, Hanoi, 12116, Vietnam.
    index: 1
date: 08 Aug 2023
bibliography: paper.bib
---


# Summary

Performance metrics are pivotal in machine learning field, especially for tasks like regression, classification, and clustering [@saura_using_2021]. They offer quantitative measures to assess the accuracy and efficacy of models, aiding researchers and practitioners in evaluating, contrasting, and enhancing algorithms and models.
In regression tasks, where continuous predictions are made, metrics such as mean squared error (MSE), root mean square error (RMSE), and Coefficient of Determination (COD) [@nguyen2018resource; @nguyen2019building] can reveal how well models capture data patterns. In classification tasks, metrics such as accuracy, precision, recall, F1-score, and AUC-ROC [@luque_impact_2019] assess a model's ability to classify instances correctly, detect false results, and gauge overall predictive performance. Clustering tasks aim to discover inherent patterns and structures within unlabeled data by grouping similar instances together. Metrics like Silhouette coefficient, Davies-Bouldin index, and Calinski-Harabasz index [@nainggolan_improved_2019] measure clustering quality, helping evaluate how well algorithms capture data distribution and assign instances to clusters.
In general, performance metrics serve multiple purposes. They enable researchers to compare different models and algorithms [@ahmed2021comprehensive], identify strengths and weaknesses [@Nguyen2019], and make informed decisions about model selection and parameter tuning [@nguyen2020new]. Moreover, it also plays a crucial role in the iterative process of model development and improvement. By quantifying the model's performance, metrics guide the optimization process [@thieu_groundwater_2023], allowing researchers to fine-tune algorithms, explore feature engineering techniques [@nguyen2021multi], and address issues such as overfitting, underfitting, and bias  [@nguyen2020eo].
This paper introduces a Python framework named **PerMetrics** (PERformance METRICS), designed to offer comprehensive performance metrics for machine learning models. The library, packaged as `permetrics`, is open-source and written in Python. It provides a wide number of metrics to enable users to evaluate their models effectively. `permetrics` is hosted on GitHub and is under continuous development and maintenance by the dedicated team. The framework is accompanied by comprehensive documentation, examples, and test cases, facilitating easy comprehension and integration into users' workflows.


# Statement of need

**PerMetrics** is a Python project developed in the field of performance assessment and machine learning. To the best of our knowledge, it is the first open-source framework that contributes a significant number of metrics, totaling 111 methods, for three fundamental problems: regression, classification, and clustering. This library relies exclusively on only two well-known third-party Python scientific computing packages: `NumPy` [@harris2020array] and `SciPy` [@virtanen2020scipy]. The modules of `permetrics` are extensively documented, and the automatically generated API provides a complete and up-to-date description of both the object-oriented and functional implementations underlying the framework.

To gain a better understanding of the necessity of **PerMetrics** library, this section will compare it to several notable libraries currently are available. Most notably, `Scikit-Learn` [@scikit_learn], which also encompasses an assortment of metrics for regression, classification, and clustering problems. Nevertheless, a few classification metrics present in `Scikit-Learn` lack support for multiple outputs, such as the Matthews correlation coefficient (MCC) and Hinge loss. Furthermore, critical metrics such as RMSE, mean absolute percentage error (MAPE), Nash-Sutcliffe efficiency (NSE), and Kling-Gupta efficiency (KGE) are absent. `permetrics` addresses these deficiencies. Additionally, `Scikit-Learn` is deficient in various vital clustering metrics, including but not limited to Ball Hall index, Banfeld Raftery index, sum of squared error, Duda Hart index, and Hartigan index [@van2023metacluster].

Another popular package is `Metrics` [@benhamner]. It provides a variety of metrics for different programming languages such as Python, MATLAB, R, and Haskell. However, the development team has ceased activity since 2015. They offer a limited number of metrics because they focused on creating a single set of metrics for multiple programming languages. Additionally, the metrics are not packaged as a complete library but rather exist as repository code on GitHub.

`TorchMetrics` [@torchmetrics] is a widely recognized framework for performance metrics developed for PyTorch users. The library includes over 100 metrics, covering various domains such as regression, classification, audio, detection, and text. However, `TorchMetrics` does not provide metrics specifically for clustering tasks. Although it offers a substantial number of metrics, it falls short compared to `permetrics`. Moreover, it relies heavily on other major libraries such as `NumPy`, `Torch`, `Typing-extensions`, `Packaging`, and `Lightning-utilities`. Additionally, using this library may not be easy for beginners in Python programming, as it requires a deep understanding of the Torch library to utilize `TorchMetrics` effectively.

Other popular libraries such as `TensorFlow` [@abadi2016tensorflow], `Keras` [@chollet2017xception], `CatBoost` [@prokhorenkova2018catboost], and `MxNet` [@chen2015mxnet] also contain modules dedicated to metrics. However, the issue with these libraries is that their metric modules are specific to each respective one. It is challenging to combine metric modules from different libraries with each other. If it is possible to combine them, it often requires installing numerous related libraries. Furthermore, the metric modules within each library are tailored to users who are familiar with that specific one, requiring users to learn multiple libraries, syntax structures, and necessary commands associated with each framework to use them in combination. These are significant obstacles when using metrics from such libraries.

All the aforementioned challenges are addressed by our **PerMetrics** library. It not only offers a simple and concise syntax and usage but also does not require any knowledge of other major libraries such as `TensorFlow`, `Keras`, or `PyTorch`. Additionally, it can be seamlessly integrated with any computational or machine learning library. In the future, we plan to expand `permetrics` to include other domains such as text metrics, audio metrics, detection metrics, and image metrics.


# Available Methods

At the time of publication, `PerMetrics` provides three types of performance metrics include regression, classification, and clustering metrics. We listed all methods of each type below.

| **Problem**    | **ID** | **Metric** | **Metric Fullname**                              |
|----------------|--------|------------|--------------------------------------------------|
| Regression     | 1      | EVS        | Explained Variance Score                         |
| ****           | 2      | ME         | Max Error                                        |
| ****           | 3      | MBE        | Mean Bias Error                                  |
| ****           | 4      | MAE        | Mean Absolute Error                              |
| ****           | 5      | MSE        | Mean Squared Error                               |
| ****           | 6      | RMSE       | Root Mean Squared Error                          |
| ****           | 7      | MSLE       | Mean Squared Log Error                           |
| ****           | 8      | MedAE      | Median Absolute Error                            |
| ****           | 9      | MRE / MRB  | Mean Relative Error / Mean Relative Bias         |
| ****           | 10     | MPE        | Mean Percentage Error                            |
| ****           | 11     | MAPE       | Mean Absolute Percentage Error                   |
| ****           | 12     | SMAPE      | Symmetric Mean Absolute Percentage Error         |
| ****           | 13     | MAAPE      | Mean Arctangent Absolute Percentage Error        |
| ****           | 14     | MASE       | Mean Absolute Scaled Error                       |
| ****           | 15     | NSE        | Nash-Sutcliffe Efficiency Coefficient            |
| ****           | 16     | NNSE       | Normalized Nash-Sutcliffe Efficiency Coefficient |
| ****           | 17     | WI         | Willmott Index                                   |
| ****           | 18     | R / PCC    | Pearson’s Correlation Coefficient                |
| ****           | 19     | AR / APCC  | Absolute Pearson's Correlation Coefficient       |
| ****           | 20     | RSQ/R2S    | (Pearson’s Correlation Index) ^ 2                |
| ****           | 21     | R2 / COD   | Coefficient of Determination                     |
| ****           | 22     | AR2 / ACOD | Adjusted Coefficient of Determination            |
| ****           | 23     | CI         | Confidence Index                                 |
| ****           | 24     | DRV        | Deviation of Runoff Volume                       |
| ****           | 25     | KGE        | Kling-Gupta Efficiency                           |
| ****           | 26     | GINI       | Gini Coefficient                                 |
| ****           | 27     | GINI_WIKI  | Gini Coefficient on Wikipage                     |
| ****           | 28     | PCD        | Prediction of Change in Direction                |
| ****           | 29     | CE         | Cross Entropy                                    |
| ****           | 30     | KLD        | Kullback Leibler Divergence                      |
| ****           | 31     | JSD        | Jensen Shannon Divergence                        |
| ****           | 32     | VAF        | Variance Accounted For                           |
| ****           | 33     | RAE        | Relative Absolute Error                          |
| ****           | 34     | A10        | A10 Index                                        |
| ****           | 35     | A20        | A20 Index                                        |
| ****           | 36     | A30        | A30 Index                                        |
| ****           | 37     | NRMSE      | Normalized Root Mean Square Error                |
| ****           | 38     | RSE        | Residual Standard Error                          |
| ****           | 39     | RE / RB    | Relative Error / Relative Bias                   |
| ****           | 40     | AE         | Absolute Error                                   |
| ****           | 41     | SE         | Squared Error                                    |
| ****           | 42     | SLE        | Squared Log Error                                |
| ****           | 43     | COV        | Covariance                                       |
| ****           | 44     | COR        | Correlation                                      |
| ****           | 45     | EC         | Efficiency Coefficient                           |
| ****           | 46     | OI         | Overall Index                                    |
| ****           | 47     | CRM        | Coefficient of Residual Mass                     |
| --             | --     | --         | --                                               |
| Classification | 1      | PS         | Precision Score                                  |
| ****           | 2      | NPV        | Negative Predictive Value                        |
| ****           | 3      | RS         | Recall Score                                     |
| ****           | 4      | AS         | Accuracy Score                                   |
| ****           | 5      | F1S        | F1 Score                                         |
| ****           | 6      | F2S        | F2 Score                                         |
| ****           | 7      | FBS        | F-Beta Score                                     |
| ****           | 8      | SS         | Specificity Score                                |
| ****           | 9      | MCC        | Matthews Correlation Coefficient                 |
| ****           | 10     | HS         | Hamming Score                                    |
| ****           | 11     | CKS        | Cohen's kappa score                              |
| ****           | 12     | JSI        | Jaccard Similarity Coefficient                   |
| ****           | 13     | GMS        | Geometric Mean Score                             |
| ****           | 14     | ROC-AUC    | ROC-AUC                                          |
| ****           | 15     | LS         | Lift Score                                       |
| ****           | 16     | GINI       | GINI Index                                       |
| ****           | 17     | CEL        | Cross Entropy Loss                               |
| ****           | 18     | HL         | Hinge Loss                                       |
| ****           | 19     | KLDL       | Kullback Leibler Divergence Loss                 |
| ****           | 20     | BSL        | Brier Score Loss                                 |
| --             | --     | --         | --                                               |
| Clustering     | 1      | BHI        | Ball Hall Index                                  |
| ****           | 2      | XBI        | Xie Beni Index                                   |
| ****           | 3      | DBI        | Davies Bouldin Index                             |
| ****           | 4      | BRI        | Banfeld Raftery Index                            |
| ****           | 5      | KDI        | Ksq Detw Index                                   |
| ****           | 6      | DRI        | Det Ratio Index                                  |
| ****           | 7      | DI         | Dunn Index                                       |
| ****           | 8      | CHI        | Calinski Harabasz Index                          |
| ****           | 9      | LDRI       | Log Det Ratio Index                              |
| ****           | 10     | LSRI       | Log SS Ratio Index                               |
| ****           | 11     | SI         | Silhouette Index                                 |
| ****           | 12     | SSEI       | Sum of Squared Error Index                       |
| ****           | 13     | MSEI       | Mean Squared Error Index                         |
| ****           | 14     | DHI        | Duda-Hart Index                                  |
| ****           | 15     | BI         | Beale Index                                      |
| ****           | 16     | RSI        | R-squared Index                                  |
| ****           | 17     | DBCVI      | Density-based Clustering Validation Index        |
| ****           | 18     | HI         | Hartigan Index                                   |
| ****           | 19     | MIS        | Mutual Info Score                                |
| ****           | 20     | NMIS       | Normalized Mutual Info Score                     |
| ****           | 21     | RaS        | Rand Score                                       |
| ****           | 22     | ARS        | Adjusted Rand Score                              |
| ****           | 23     | FMS        | Fowlkes Mallows Score                            |
| ****           | 24     | HS         | Homogeneity Score                                |
| ****           | 25     | CS         | Completeness Score                               |
| ****           | 26     | VMS        | V-Measure Score                                  |
| ****           | 27     | PrS        | Precision Score                                  |
| ****           | 28     | ReS        | Recall Score                                     |
| ****           | 29     | FmS        | F-Measure Score                                  |
| ****           | 30     | CDS        | Czekanowski Dice Score                           |
| ****           | 31     | HGS        | Hubert Gamma Score                               |
| ****           | 32     | JS         | Jaccard Score                                    |
| ****           | 33     | KS         | Kulczynski Score                                 |
| ****           | 34     | MNS        | Mc Nemar Score                                   |
| ****           | 35     | PhS        | Phi Score                                        |
| ****           | 36     | RTS        | Rogers Tanimoto Score                            |
| ****           | 37     | RRS        | Russel Rao Score                                 |
| ****           | 38     | SS1S       | Sokal Sneath1 Score                              |
| ****           | 39     | SS2S       | Sokal Sneath2 Score                              |
| ****           | 40     | PuS        | Purity Score                                     |
| ****           | 41     | ES         | Entropy Score                                    |
| ****           | 42     | TS         | Tau Score                                        |
| ****           | 43     | GAS        | Gamma Score                                      |
| ****           | 44     | GPS        | Gplus Score                                      |


# Installation and Simple Example

**PerMetrics** is [published](https://pypi.org/project/permetrics/) to the Python Packaging Index (PyPI) and can be installed via pip

```bash
pip install permetrics
```

Below are a few fundamental examples illustrating the usage of the `permetrics` library. We have prepared a folder `examples` in Github repository that contains these examples and more advances one. Furthermore, to gain a comprehensive understanding of our library, we recommend reading the documentation available at the following [link](https://permetrics.readthedocs.io/).

## Regression Metrics

```python
import numpy as np
from permetrics import RegressionMetric

y_true = np.array([3, -0.5, 2, 7, 5, 6])
y_pred = np.array([2.5, 0.0, 2, 8, 5, 6])

evaluator = RegressionMetric(y_true=y_true, y_pred=y_pred)

print(evaluator.mean_squared_error())
print(evaluator.median_absolute_error())
print(evaluator.MAPE())
```

## Classification Metrics

```python
from permetrics import ClassificationMetric

## For integer labels or categorical labels
y_true = [0, 1, 0, 0, 1, 0]
y_pred = [0, 1, 0, 0, 0, 1]
# y_true = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]

evaluator = ClassificationMetric(y_true, y_pred)

print(evaluator.f1_score())
print(evaluator.F1S(average="micro"))
print(evaluator.f1_score(average="macro"))
```

## Clustering Metrics

```python
import numpy as np
from permetrics import ClusteringMetric
from sklearn.datasets import make_blobs

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
y_pred = np.random.randint(0, 4, size=300)

evaluator = ClusteringMetric(y_true=y_true, y_pred=y_pred, X=X)

# Call specific function inside evaluator, each function has 2 names 
#   (fullname and short name)

## 1. Internal metrics: Need X and y_pred and has function's suffix as `index`
print(evaluator.ball_hall_index(X=X, y_pred=y_pred))
print(evaluator.CHI(X=X, y_pred=y_pred))

## 2. External metrics: Need y_true and y_pred and has function's suffix as `score`
print(evaluator.adjusted_rand_score(y_true=y_true, y_pred=y_pred))
print(evaluator.completeness_score(y_true=y_true, y_pred=y_pred))
```


# Acknowledgements
We express our sincere thanks to the individuals who have enhanced our software through their valuable issue reports and insightful feedback.


# References
