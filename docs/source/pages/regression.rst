Regression Metrics
==================

* Metrics Table

================= ====== ============ =================================================== ======================================================
Problem           STT    Metric       Metric Fullname                                     Characteristics
================= ====== ============ =================================================== ======================================================
Regression        1      EVS          Explained Variance Score                            Greater is better (Best = 1), Range=(-inf, 1.0]
                2      ME           Max Error                                           Smaller is better (Best = 0), Range=[0, +inf)
                3      MAE          Mean Absolute Error                                 Smaller is better (Best = 0), Range=[0, +inf)
                4      MSE          Mean Squared Error                                  Smaller is better (Best = 0), Range=[0, +inf)
                5      RMSE         Root Mean Squared Error                             Smaller is better (Best = 0), Range=[0, +inf)
                6      MSLE         Mean Squared Log Error                              Smaller is better (Best = 0), Range=[0, +inf)
                7      MedAE        Median Absolute Error                               Smaller is better (Best = 0), Range=[0, +inf)
                8      MRE          Mean Relative Error                                 Smaller is better (Best = 0), Range=[0, +inf)
                9      MAPE         Mean Absolute Percentage Error                      Smaller is better (Best = 0), Range=[0, +inf)
                10     SMAPE        Symmetric Mean Absolute Percentage Error            Smaller is better (Best = 0), Range=[0, 1]
                11     MAAPE        Mean Arctangent Absolute Percentage Error           Smaller is better (Best = 0), Range=[0, +inf)
                12     MASE         Mean Absolute Scaled Error                          Smaller is better (Best = 0), Range=[0, +inf)
                13     NSE          Nash-Sutcliffe Efficiency Coefficient               Greater is better (Best = 1), Range=(-inf, 1]
                14     NNSE         Normalized Nash-Sutcliffe Efficiency Coefficient    Greater is better (Best = 1), Range=[0, 1]
                15     WI           Willmott Index                                      Greater is better (Best = 1), Range=[0, 1]
                16     R/PCC        Pearson’s Correlation Coefficient                   Greater is better (Best = 1), Range=[-1, 1]
                17     R2s          (Pearson’s Correlation Index) ^ 2                   Greater is better (Best = 1), Range=[0, 1]
                18     R2           Coefficient of Determination                        Greater is better (Best = 1), Range=(-inf, 1]
                19     CI           Confidence Index                                    Greater is better (Best = 1), Range=(-inf, 1]
                20     DRV           Deviation of Runoff Volume                         Smaller is better (Best = 1.0), Range=[1, +inf)
                21     KGE          Kling-Gupta Efficiency                              Greater is better (Best = 1), Range=(-inf, 1]
                22     GINI         Gini Coefficient                                    Smaller is better (Best = 0), Range=[0, +inf)
                23     GINI_WIKI    Gini Coefficient in Wiki                            Smaller is better (Best = 0), Range=[0, +inf)
                24     PCD          Prediction of Change in Direction                   Greater is better (Best = 1.0), Range=[0, 1]
                25     CE           Cross Entropy                                       Range(-inf, 0], Can't give comment about this
                26     KLD          Kullback Leibler Divergence                         Best = 0, Range=(-inf, +inf)
                27     JSD          Jensen Shannon Divergence                           Smaller is better (Best = 0), Range=[0, +inf)
                28     VAF          Variance Accounted For                              Greater is better (Best = 100%), Range=(-inf, 100%]
                29     RAE          Relative Absolute Error                             Smaller is better (Best = 0), Range=[0, +inf)
                30     A10          A10 Index                                           Greater is better (Best = 1), Range=[0, 1]
                31     A20          A20 Index                                           Greater is better (Best = 1), Range=[0, 1]
                32     NRMSE        Normalized Root Mean Square Error                   Smaller is better (Best = 0), Range=[0, +inf)
                33     RSE          Residual Standard Error                             Smaller is better (Best = 0), Range=[0, +inf)
                34     RE           Relative error                                      Best = 0, Range=(-inf, +inf)
                35     AE           Absolute error                                      Best = 0, Range=(-inf, +inf)
                36     SE            Squared error                                      Smaller is better (Best = 0), Range=[0, +inf)
                37     SLE          Squared log error                                   Smaller is better (Best = 0), Range=[0, +inf)
                38
Classification    1      MLL          Mean Log Likelihood
                2      LL           Log Likelihood
================= ====== ============ ===================================================



From now on:

+ :math:`\hat{y}` is the estimated target output,
+ :math:`y` is the corresponding (correct) target output.
+ :math:`\hat{Y}` is the whole estimated target output ,
+ :math:`Y` is the corresponding (correct) target output.
+ :math:`mean(\hat{Y})` is the mean of whole estimated target output ,
+ :math:`mean(Y)` is the mean of whole (correct) target output.




.. toctree::
   :maxdepth: 3

   regression/EVS.rst
   regression/ME.rst
   regression/MAE.rst
   regression/MSE.rst
   regression/RMSE.rst
   regression/MSLE.rst
   regression/MedAE.rst
   regression/MRE.rst
   regression/MAPE.rst
   regression/SMAPE.rst
   regression/MAAPE.rst
   regression/MASE.rst
   regression/NSE.rst
   regression/NNSE.rst
   regression/WI.rst
   regression/R.rst
   regression/R2.rst
   regression/CI.rst
   regression/R2s.rst
   regression/DRV.rst
   regression/KGE.rst

   regression/GINI.rst
   regression/PCD.rst
   regression/CE.rst
   regression/KLD.rst
   regression/JSD.rst
   regression/VAF.rst
   regression/RAE.rst
   regression/A10.rst
   regression/A20.rst
   regression/NRMSE.rst
   regression/RSE.rst

   regression/RE.rst
   regression/AE.rst
   regression/SE.rst
   regression/SLE.rst

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3
