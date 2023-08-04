Regression Metrics
==================

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
   regression/MBE.rst
   regression/RMSE.rst
   regression/MSLE.rst
   regression/MedAE.rst
   regression/MRE.rst
   regression/MPE.rst
   regression/MAPE.rst
   regression/SMAPE.rst
   regression/MAAPE.rst
   regression/MASE.rst
   regression/NSE.rst
   regression/NNSE.rst
   regression/WI.rst
   regression/R.rst
   regression/AR.rst
   regression/R2.rst
   regression/AR2.rst
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
   regression/A30.rst
   regression/NRMSE.rst
   regression/RSE.rst
   regression/COV.rst
   regression/COR.rst
   regression/EC.rst
   regression/OI.rst
   regression/CRM.rst

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
