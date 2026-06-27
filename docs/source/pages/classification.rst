Classification Metrics
======================

.. toctree::
   :maxdepth: 1

   classification/AS.rst
   classification/CKS.rst
   classification/F1S.rst
   classification/F2S.rst
   classification/FBS.rst
   classification/GINI.rst
   classification/GMS.rst
   classification/PS.rst
   classification/NPV.rst
   classification/RS.rst
   classification/SS.rst
   classification/MCC.rst
   classification/ROC_AUC.rst
   classification/LS.rst
   classification/HML.rst
   classification/HGL.rst
   classification/JSI.rst
   classification/KLDL.rst
   classification/BSL.rst
   classification/CEL.rst



==========================
All classification metrics
==========================

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
| 9   | MCC     | Matthews Correlation Coefficient | Bigger is better (Best = 1), Range = [-1, 1]        |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 11  | CKS     | Cohen's kappa score              | Bigger is better (Best = 1), Range = [-1, 1]        |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 12  | JSI/JSS | Jaccard Similarity Score         | Bigger is better (Best = 1), Range = [0, 1]         |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 13  | GMS     | Geometric Mean Score             | Bigger is better (Best = 1), Range = [0, 1]         |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 14  | ROC/AUC | ROC / AUC / RAS                  | Bigger is better (Best = 1), Range = [0, 1]         |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 15  | LS      | Lift Score                       | Bigger is better (Unknown), Range = [0, +inf)       |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 16  | GINI    | GINI Index                       | Bigger is better (Best = 1), Range = [-1, 1]        |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 17  | CEL     | Cross Entropy Loss               | Smaller is better (Best = 0), Range=[0, +inf)       |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 10  | HML     | Hamming Loss                     | Smaller is better (Best = 0), Range = [0, 1]        |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 18  | HGL     | Hinge Loss                       | Smaller is better (Best = 0), Range=[0, +inf)       |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 19  | KLDL    | Kullback Leibler Divergence Loss | Smaller is better (Best = 0), Range=[0, +inf)       |
+-----+---------+----------------------------------+-----------------------------------------------------+
| 20  | BSL     | Brier Score Loss                 | Smaller is better (Best = 0), Range=[0, 1]          |
+-----+---------+----------------------------------+-----------------------------------------------------+
