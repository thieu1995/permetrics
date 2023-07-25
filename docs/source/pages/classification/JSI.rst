Jaccard Similarity Index (JSI)
==============================

.. toctree::
   :maxdepth: 3
   :caption: Jaccard Similarity Index (JSI)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


+ Best possible score is 1.0, higher value is better. Range = [0, 1]

The Jaccard similarity index, also known as the Jaccard similarity coefficient or Jaccard index, is a commonly used evaluation metric in binary and
multiclass classification problems. It measures the similarity between the predicted labels y_pred and the true labels y_true, and is defined
as the ratio of the number of true positive (TP) predictions to the number of true positive and false positive (FP) predictions.

In a binary classification problem, a prediction is considered true positive if both the true label and the predicted label are positive,
and false positive if the true label is negative and the predicted label is positive. False negative (FN) predictions are those where the
true label is positive and the predicted label is negative. True negative (TN) predictions are those where both the true label and
the predicted label are negative.

To calculate the Jaccard similarity index, the formula is::

	J = TP / (TP + FP)

Where TP (True Positives) is the number of instances where the true label and the predicted label are both positive, and FP (False Positives)
is the number of instances where the true label is negative and the predicted label is positive.

In a multiclass classification problem, the Jaccard similarity index can be calculated for each class individually, and then averaged
over all classes to obtain the overall Jaccard similarity index. The weighted average of the Jaccard similarity indices can also be calculated,
with the weights given by the number of instances in each class.


The Jaccard similarity index ranges from 0 to 1, with a value of 1 indicating perfect agreement between the predicted labels and the true labels,
and a value of 0 indicating complete disagreement. The Jaccard similarity index is a useful evaluation metric in situations where the class
distribution is imbalanced, as it takes into account only true positive and true negative predictions and not false positive or false negative predictions.

It's important to note that the Jaccard similarity index is sensitive to the number of instances in each class, so it's recommended to
use this metric in combination with other evaluation metrics, such as precision, recall, and F1-score, to get a complete picture of
the performance of a classification model.




Example:

.. code-block:: python
	:emphasize-lines: 11,13-16

	from numpy import array
	from permetrics.classification import ClassificationMetric

	## For integer labels or categorical labels
	y_true = [0, 1, 0, 0, 1, 0]
	y_pred = [0, 1, 0, 0, 0, 1]

	# y_true = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
	# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]

	cm = ClassificationMetric(y_true, y_pred, decimal = 5)

	print(cm.jaccard_similarity_index(average=None))
	print(cm.jaccard_similarity_coefficient(average="micro"))
	print(cm.jsi(average="macro"))
	print(cm.jsc(average="weighted"))

