Classification Metrics
======================


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



In extending a binary metric to multiclass or multilabel problems, the data is treated as a collection of binary problems, one for each class.
There are then a number of ways to average binary metric calculations across the set of classes, each of which may be useful in some scenario.
Where available, you should select among these using the average parameter.

+ "micro" gives each sample-class pair an equal contribution to the overall metric (except as a result of sample-weight). Rather than summing the metric per
class, this sums the dividends and divisors that make up the per-class metrics to calculate an overall quotient.  Micro-averaging may be preferred in
multilabel settings, including multiclass classification where a majority class is to be ignored.  Calculate metrics globally by considering each element of
the label indicator matrix as a label.

+ "macro" simply calculates the mean of the binary metrics, giving equal weight to each class. In problems where infrequent classes are nonetheless important,
macro-averaging may be a means of highlighting their performance. On the other hand, the assumption that all classes are equally
important is often untrue, such that macro-averaging will over-emphasize the typically low performance on an infrequent class.

+ "weighted" accounts for class imbalance by computing the average of binary metrics in which each class’s score is weighted by its presence in the true data sample.

+ None: will return an array with the score for each class.



.. toctree::
   :maxdepth: 3

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
   classification/HS.rst
   classification/LS.rst
   classification/JSI.rst
   classification/ROC-AUC.rst

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3
