Classification Metrics
======================

+------+----------+-----------------------------------+------------------------------------------------+
| STT  | Metric   | Metric Fullname                   | Characteristics                                |
+======+==========+===================================+================================================+
| 1    | PS       | Precision Score                   | Higher is better (Best = 1), Range = [0, 1]    |
+------+----------+-----------------------------------+------------------------------------------------+
| 2    | NPV      | Negative Predictive Value         | Higher is better (Best = 1), Range = [0, 1]    |
+------+----------+-----------------------------------+------------------------------------------------+
| 3    | RS       | Recall Score                      | Higher is better (Best = 1), Range = [0, 1]    |
+------+----------+-----------------------------------+------------------------------------------------+
| 4    | AS       | Accuracy Score                    | Higher is better (Best = 1), Range = [0, 1]    |
+------+----------+-----------------------------------+------------------------------------------------+
| 5    | F1S      | F1 Score                          | Higher is better (Best = 1), Range = [0, 1]    |
+------+----------+-----------------------------------+------------------------------------------------+
| 6    | F2S      | F2 Score                          | Higher is better (Best = 1), Range = [0, 1]    |
+------+----------+-----------------------------------+------------------------------------------------+
| 7    | FBS      | F-Beta Score                      | Higher is better (Best = 1), Range = [0, 1]    |
+------+----------+-----------------------------------+------------------------------------------------+
| 8    | SS       | Specificity Score                 | Higher is better (Best = 1), Range = [0, 1]    |
+------+----------+-----------------------------------+------------------------------------------------+
| 9    | MCC      | Matthews Correlation Coefficient  | Higher is better (Best = 1), Range = [-1, +1]  |
+------+----------+-----------------------------------+------------------------------------------------+
| 10   | HL       | Hamming Loss                      | Higher is better (Best = 1), Range = [0, 1]    |
+------+----------+-----------------------------------+------------------------------------------------+
| 11   | LS       | Lift Score                        | Higher is better (Best=+inf), Range = [0, +inf)|
+------+----------+-----------------------------------+------------------------------------------------+


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

   classification/PS.rst
   classification/NPV.rst
   classification/RS.rst
   classification/AS.rst
   classification/F1S.rst
   classification/F2S.rst
   classification/FBS.rst
   classification/SS.rst
   classification/MCC.rst
   classification/HL.rst
   classification/LS.rst
   classification/CKS.rst

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3
