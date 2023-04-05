GINI Index
==========

.. toctree::
   :maxdepth: 3
   :caption: GINI Index (GINI)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


The Gini index is also used as a metric to evaluate the performance of a binary classification model. It is a measure of how well the model separates the
positive and negative classes.

The Gini index is calculated as follows:

+ Sort the predicted probabilities of the positive class in descending order.
+ Calculate the cumulative sum of the true positive rate (TPR) and false positive rate (FPR) at each threshold, where TPR is the proportion of positive
samples correctly classified as positive, and FPR is the proportion of negative samples incorrectly classified as positive.
+ Calculate the area under the curve (AUC) of the cumulative sum, and multiply it by 2.


The resulting value ranges from 0 to 1, where 0 indicates that the model predicts all negative samples as positive and 1 indicates perfect separation between the positive and negative classes.

The Gini index can be used as an alternative to the AUC-ROC metric, and it has some advantages in terms of interpretation and sensitivity to class imbalance. However, it can be less commonly used in practice, and the AUC-ROC is often preferred as a metric for binary classification.

+ Best possible score is 1.0, higher value is better. Range = [0, 1]
+ There is no "micro" average mode in GINI index metric


Example:

.. code-block:: python
	:emphasize-lines: 11-14

	from numpy import array
	from permetrics.classification import ClassificationMetric

	## For integer labels or categorical labels
	y_true = [0, 1, 0, 0, 1, 0]
	y_pred = [0, 1, 0, 0, 0, 1]

	# y_true = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
	# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]

	cm = ClassificationMetric(y_true, y_pred, decimal=5)
	print(cm.gini_index(average=None))
	print(cm.GINI(average="macro"))
	print(cm.gini(average="weighted"))
