ROC-AUC
=======

.. toctree::
   :maxdepth: 3
   :caption: ROC-AUC

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. image:: /_static/images/class_score_1.png

ROC-AUC (Receiver Operating Characteristic - Area Under the Curve) is a metric used to evaluate the performance of a binary classification model. It is a measure of how well the model is able to distinguish between positive and negative classes.

A ROC curve is a plot of the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings. The TPR is the ratio of the number of true positives to the total number of positives, while the FPR is the ratio of the number of false positives to the total number of negatives.

The AUC is the area under the ROC curve. It ranges between 0 and 1, where a value of 0.5 represents a model that performs no better than random guessing, and a value of 1 represents a model that makes perfect predictions. A higher AUC value indicates that the model is better at distinguishing between the positive and negative classes.

Interpretation of the ROC curve and AUC value depends on the specific problem and domain. In general, a model with an AUC value of 0.7 to 0.8 is considered acceptable, while a value greater than 0.8 is considered good. However, the interpretation may vary depending on the specific use case and the cost of false positives and false negatives.


In the multi-class and multi-label case, this is the average of the AS score of each class with weighting depending on the average parameter.

In a multiclass classification problem, ROC-AUC can still be used as a metric, but it requires some modifications to account for the multiple classes.

One approach is to use the one-vs-all (OvA) strategy, where we train a binary classifier for each class, treating it as the positive class and all other classes as the negative class. For each class, we calculate the ROC curve and AUC value, and then average the AUC values across all classes to obtain a single metric.

Another approach is to use the one-vs-one (OvO) strategy, where we train a binary classifier for each pair of classes, treating one class as the positive class and the other as the negative class. For each pair of classes, we calculate the ROC curve and AUC value, and then average the AUC values across all pairs to obtain a single metric.

In either case, it is important to ensure that the classes are balanced, meaning that the number of examples in each class is roughly equal, or to use appropriate sampling techniques to handle class imbalance.

It is worth noting that ROC-AUC may not always be the best metric for multiclass problems, especially when the classes are highly imbalanced or the cost of false positives and false negatives varies across classes. In such cases, other metrics such as precision, recall, F1-score, or weighted average of these metrics may be more appropriate.

+ Best possible score is 1.0, higher value is better. Range = [0, 1]
+ https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/
+ There is no "micro" average mode in ROC-AUC metric

Example:

.. code-block:: python
	:emphasize-lines: 20-23

	from numpy import array
	from permetrics.classification import ClassificationMetric

	## For integer labels or categorical labels
	y_true = [0, 1, 0, 0, 1, 0]
	y_score = [0, 1, 0, 0, 0, 1]

	y_true = np.array([0, 1, 2, 1, 2, 0, 0, 1])
	y_score = np.array([[0.8, 0.1, 0.1],
                   [0.2, 0.5, 0.3],
                   [0.1, 0.3, 0.6],
                   [0.3, 0.7, 0.0],
                   [0.4, 0.3, 0.3],
                   [0.6, 0.2, 0.2],
                   [0.9, 0.1, 0.0],
                   [0.1, 0.8, 0.1]])

	cm = ClassificationMetric(y_true, y_pred, decimal = 5)

	print(cm.roc_auc_score(y_true, y_score, average=None))
	print(cm.ROC(y_true, y_score))
	print(cm.AUC(y_true, y_score, average="macro"))
	print(cm.RAS(y_true, y_score, average="weighted"))
