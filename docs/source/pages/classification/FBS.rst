F-Beta Score (FBS)
==================

.. toctree::
   :maxdepth: 3
   :caption: F-Beta Score (FBS)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


The F-beta score is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 and its worst value at 0.

The beta parameter determines the weight of recall in the combined score. beta < 1 lends more weight to precision,
while beta > 1 favors recall (beta -> 0 considers only precision, beta -> +inf only recall).::

	F-beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

In the multi-class and multi-label case, this is the average of the FBS score of each class with weighting depending on the average parameter.

+ Best possible score is 1.0, higher value is better. Range = [0, 1]
+ https://neptune.ai/blog/evaluation-metrics-binary-classification
+ https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html#sklearn.metrics.fbeta_score


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

	print(cm.fbeta_score())
	print(cm.fbeta_score(average="micro"))
	print(cm.fbs(average="macro"))
	print(cm.fbs(average="weighted"))

