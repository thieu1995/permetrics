================
Citation Request
================

Please include these citations if you plan to use this library::

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

If you have an open-ended or a research question, you can contact me via nguyenthieu2102@gmail.com


=======================
All Performance Metrics
=======================

The list of all available performance metrics in this library are as follows:

+-----+------------+--------------------------------------------------+------------------------------------------------------+
| STT |   Metric   |                 Metric Fullname                  |                   Characteristics                    |
+=====+============+==================================================+======================================================+
|  1  |    EVS     |             Explained Variance Score             |    Bigger is better (Best = 1), Range=(-inf, 1.0]    |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
|  2  |     ME     |                    Max Error                     |    Smaller is better (Best = 0), Range=[0, +inf)     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
|  3  |    MBE     |                 Mean Bias Error                  |             Best = 0, Range=(-inf, +inf)             |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
|  4  |    MAE     |               Mean Absolute Error                |    Smaller is better (Best = 0), Range=[0, +inf)     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
|  5  |    MSE     |                Mean Squared Error                |    Smaller is better (Best = 0), Range=[0, +inf)     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
|  6  |    RMSE    |             Root Mean Squared Error              |    Smaller is better (Best = 0), Range=[0, +inf)     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
|  7  |    MSLE    |              Mean Squared Log Error              |    Smaller is better (Best = 0), Range=[0, +inf)     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
|  8  |   MedAE    |              Median Absolute Error               |    Smaller is better (Best = 0), Range=[0, +inf)     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
|  9  | MRE / MRB  |     Mean Relative Error / Mean Relative Bias     |    Smaller is better (Best = 0), Range=[0, +inf)     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 10  |    MPE     |              Mean Percentage Error               |             Best = 0, Range=(-inf, +inf)             |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 11  |    MAPE    |          Mean Absolute Percentage Error          |    Smaller is better (Best = 0), Range=[0, +inf)     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 12  |   SMAPE    |     Symmetric Mean Absolute Percentage Error     |      Smaller is better (Best = 0), Range=[0, 1]      |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 13  |   MAAPE    |    Mean Arctangent Absolute Percentage Error     |    Smaller is better (Best = 0), Range=[0, +inf)     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 14  |    MASE    |            Mean Absolute Scaled Error            |    Smaller is better (Best = 0), Range=[0, +inf)     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 15  |    NSE     |      Nash-Sutcliffe Efficiency Coefficient       |     Bigger is better (Best = 1), Range=(-inf, 1]     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 16  |    NNSE    | Normalized Nash-Sutcliffe Efficiency Coefficient |      Bigger is better (Best = 1), Range=[0, 1]       |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 17  |     WI     |                  Willmott Index                  |      Bigger is better (Best = 1), Range=[0, 1]       |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 18  |  R / PCC   |        Pearson’s Correlation Coefficient         |      Bigger is better (Best = 1), Range=[-1, 1]      |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 19  | AR / APCC  |    Absolute Pearson's Correlation Coefficient    |      Bigger is better (Best = 1), Range=[-1, 1]      |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 20  |  RSQ/R2S   |        (Pearson’s Correlation Index) ^ 2         |      Bigger is better (Best = 1), Range=[0, 1]       |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 21  |  R2 / COD  |           Coefficient of Determination           |     Bigger is better (Best = 1), Range=(-inf, 1]     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 22  | AR2 / ACOD |      Adjusted Coefficient of Determination       |     Bigger is better (Best = 1), Range=(-inf, 1]     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 23  |     CI     |                 Confidence Index                 |     Bigger is better (Best = 1), Range=(-inf, 1]     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 24  |    DRV     |            Deviation of Runoff Volume            |   Smaller is better (Best = 1.0), Range=[1, +inf)    |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 25  |    KGE     |              Kling-Gupta Efficiency              |     Bigger is better (Best = 1), Range=(-inf, 1]     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 26  |    GINI    |                 Gini Coefficient                 |    Smaller is better (Best = 0), Range=[0, +inf)     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 27  | GINI_WIKI  |           Gini Coefficient on Wikipage           |    Smaller is better (Best = 0), Range=[0, +inf)     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 28  |    PCD     |        Prediction of Change in Direction         |     Bigger is better (Best = 1.0), Range=[0, 1]      |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 29  |     CE     |                  Cross Entropy                   |    Range(-inf, 0], Can't give comment about this     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 30  |    KLD     |           Kullback Leibler Divergence            |             Best = 0, Range=(-inf, +inf)             |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 31  |    JSD     |            Jensen Shannon Divergence             |    Smaller is better (Best = 0), Range=[0, +inf)     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 32  |    VAF     |              Variance Accounted For              |  Bigger is better (Best = 100%), Range=(-inf, 100%]  |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 33  |    RAE     |             Relative Absolute Error              |    Smaller is better (Best = 0), Range=[0, +inf)     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 34  |    A10     |                    A10 Index                     |      Bigger is better (Best = 1), Range=[0, 1]       |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 35  |    A20     |                    A20 Index                     |      Bigger is better (Best = 1), Range=[0, 1]       |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 36  |    A30     |                    A30 Index                     |      Bigger is better (Best = 1), Range=[0, 1]       |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 37  |   NRMSE    |        Normalized Root Mean Square Error         |    Smaller is better (Best = 0), Range=[0, +inf)     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 38  |    RSE     |             Residual Standard Error              |    Smaller is better (Best = 0), Range=[0, +inf)     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 39  |  RE / RB   |          Relative Error / Relative Bias          |             Best = 0, Range=(-inf, +inf)             |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 40  |     AE     |                  Absolute Error                  |             Best = 0, Range=(-inf, +inf)             |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 41  |     SE     |                  Squared Error                   |    Smaller is better (Best = 0), Range=[0, +inf)     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 42  |    SLE     |                Squared Log Error                 |    Smaller is better (Best = 0), Range=[0, +inf)     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 43  |    COV     |                    Covariance                    | Bigger is better (No best value), Range=(-inf, +inf) |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 44  |    COR     |                   Correlation                    |     Bigger is better (Best = 1), Range=[-1, +1]      |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 45  |     EC     |              Efficiency Coefficient              |    Bigger is better (Best = 1), Range=(-inf, +1]     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 46  |     OI     |                  Overall Index                   |    Bigger is better (Best = 1), Range=(-inf, +1]     |
+-----+------------+--------------------------------------------------+------------------------------------------------------+
| 47  |    CRM     |           Coefficient of Residual Mass           |   Smaller is better (Best = 0), Range=(-inf, +inf)   |
+-----+------------+--------------------------------------------------+------------------------------------------------------+



+-----+---------+----------------------------------+-----------------------------------------------------+
| STT | Metric  | Metric Fullname                  | Characteristics                                     |
+=====+=========+==================================+=====================================================+
| 1   | PS      | Precision Score                  | Bigger is better (Best = 1), Range = [0, 1]         |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 2   | NPV     | Negative Predictive Value        | Bigger is better (Best = 1), Range = [0, 1]         |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 3   | RS      | Recall Score                     | Bigger is better (Best = 1), Range = [0, 1]         |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 4   | AS      | Accuracy Score                   | Bigger is better (Best = 1), Range = [0, 1]         |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 5   | F1S     | F1 Score                         | Bigger is better (Best = 1), Range = [0, 1]         |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 6   | F2S     | F2 Score                         | Bigger is better (Best = 1), Range = [0, 1]         |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 7   | FBS     | F-Beta Score                     | Bigger is better (Best = 1), Range = [0, 1]         |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 8   | SS      | Specificity Score                | Bigger is better (Best = 1), Range = [0, 1]         |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 9   | MCC     | Matthews Correlation Coefficient | Bigger is better (Best = 1), Range = [-1, +1]       |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 10  | HS      | Hamming Score                    | Bigger is better (Best = 1), Range = [0, 1]         |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 11  | CKS     | Cohen's kappa score              | Bigger is better (Best = +1), Range = [-1, +1]      |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 12  | JSI     | Jaccard Similarity Coefficient   | Bigger is better (Best = +1), Range = [0, +1]       |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 13  | GMS     | Geometric Mean Score             | Bigger is better (Best = +1), Range = [0, +1]       |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 14  | ROC-AUC | ROC-AUC                          | Bigger is better (Best = +1), Range = [0, +1]       |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 15  | LS      | Lift Score                       | Bigger is better (No best value), Range = [0, +inf) |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 16  | GINI    | GINI Index                       | Smaller is better (Best = 0), Range = [0, +1]       |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 17  | CEL     | Cross Entropy Loss               | Smaller is better (Best = 0), Range=[0, +inf)       |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 18  | HL      | Hinge Loss                       | Smaller is better (Best = 0), Range=[0, +inf)       |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 19  | KLDL    | Kullback Leibler Divergence Loss | Smaller is better (Best = 0), Range=[0, +inf)       |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 20  | BSL     | Brier Score Loss                 | Smaller is better (Best = 0), Range=[0, +1]         |
+-----+---------+----------------------------------+-----------------------------------------------------+



+-----+--------+-------------------------------------------+--------------------------------------------------------+
| STT | Metric | Metric Fullname                           | Characteristics                                        |
+=====+========+===========================================+========================================================+
| 1   | BHI    | Ball Hall Index                           | Smaller is better (Best = 0), Range=[0, +inf)          |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 2   | XBI    | Xie Beni Index                            | Smaller is better (Best = 0), Range=[0, +inf)          |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 3   | DBI    | Davies Bouldin Index                      | Smaller is better (Best = 0), Range=[0, +inf)          |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 4   | BRI    | Banfeld Raftery Index                     | Smaller is better (No best value), Range=(-inf, inf)   |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 5   | KDI    | Ksq Detw Index                            | Smaller is better (No best value), Range=(-inf, +inf)  |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 6   | DRI    | Det Ratio Index                           | Bigger is better (No best value), Range=[0, +inf)      |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 7   | DI     | Dunn Index                                | Bigger is better (No best value), Range=[0, +inf)      |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 8   | CHI    | Calinski Harabasz Index                   | Bigger is better (No best value), Range=[0, inf)       |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 9   | LDRI   | Log Det Ratio Index                       | Bigger is better (No best value), Range=(-inf, +inf)   |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 10  | LSRI   | Log SS Ratio Index                        | Bigger is better (No best value), Range=(-inf, +inf)   |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 11  | SI     | Silhouette Index                          | Bigger is better (Best = 1), Range = [-1, +1]          |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 12  | SSEI   | Sum of Squared Error Index                | Smaller is better (Best = 0), Range = [0, +inf)        |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 13  | MSEI   | Mean Squared Error Index                  | Smaller is better (Best = 0), Range = [0, +inf)        |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 14  | DHI    | Duda-Hart Index                           | Smaller is better (Best = 0), Range = [0, +inf)        |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 15  | BI     | Beale Index                               | Smaller is better (Best = 0), Range = [0, +inf)        |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 16  | RSI    | R-squared Index                           | Bigger is better (Best=1), Range = (-inf, 1]           |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 17  | DBCVI  | Density-based Clustering Validation Index | Bigger is better (Best=0), Range = [0, 1]              |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 18  | HI     | Hartigan Index                            | Bigger is better (best=0), Range = [0, +inf)           |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 19  | MIS    | Mutual Info Score                         | Bigger is better (No best value), Range = [0, +inf)    |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 20  | NMIS   | Normalized Mutual Info Score              | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 21  | RaS    | Rand Score                                | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 22  | ARS    | Adjusted Rand Score                       | Bigger is better (Best = 1), Range = [-1, 1]           |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 23  | FMS    | Fowlkes Mallows Score                     | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 24  | HS     | Homogeneity Score                         | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 25  | CS     | Completeness Score                        | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 26  | VMS    | V-Measure Score                           | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 27  | PrS    | Precision Score                           | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 28  | ReS    | Recall Score                              | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 29  | FmS    | F-Measure Score                           | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 30  | CDS    | Czekanowski Dice Score                    | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 31  | HGS    | Hubert Gamma Score                        | Bigger is better (Best = 1), Range=[-1, +1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 32  | JS     | Jaccard Score                             | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 33  | KS     | Kulczynski Score                          | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 34  | MNS    | Mc Nemar Score                            | Bigger is better (No best value), Range=(-inf, +inf)   |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 35  | PhS    | Phi Score                                 | Bigger is better (No best value), Range = (-inf, +inf) |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 36  | RTS    | Rogers Tanimoto Score                     | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 37  | RRS    | Russel Rao Score                          | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 38  | SS1S   | Sokal Sneath1 Score                       | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 39  | SS2S   | Sokal Sneath2 Score                       | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 40  | PuS    | Purity Score                              | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 41  | ES     | Entropy Score                             | Smaller is better (Best = 0), Range = [0, +inf)        |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 42  | TS     | Tau Score                                 | Bigger is better (No best value), Range = (-inf, +inf) |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 43  | GAS    | Gamma Score                               | Bigger is better (Best = 1), Range = [-1, 1]           |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 44  | GPS    | Gplus Score                               | Smaller is better (Best = 0), Range = [0, 1]           |
+-----+--------+-------------------------------------------+--------------------------------------------------------+




==============
Official Links
==============

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


===================
Reference Documents
===================

1) https://www.debadityachakravorty.com/ai-ml/cmatrix/
2) https://neptune.ai/blog/evaluation-metrics-binary-classification
3) https://danielyang1009.github.io/model-performance-measure/
4) https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
5) http://cran.nexr.com/web/packages/clusterCrit/vignettes/clusterCrit.pdf
6) https://publikationen.bibliothek.kit.edu/1000120412/79692380
7) https://torchmetrics.readthedocs.io/en/latest/
8) http://rasbt.github.io/mlxtend/user_guide/evaluate/lift_score/
9) https://www.baeldung.com/cs/multi-class-f1-score
10) https://kavita-ganesan.com/how-to-compute-precision-and-recall-for-a-multi-class-classification-problem/#.YoXMSqhBy3A
11) https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/



=======
License
=======

The project is licensed under GNU General Public License (GPL) V3 license.


.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
