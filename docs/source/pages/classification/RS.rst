Recall Score (RS)
=================

.. toctree::
   :maxdepth: 3
   :caption: Recall Score (RS)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. image:: /_static/images/class_score_1.png

The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives.
The recall is intuitively the ability of the classifier to find all the positive samples.

In the multi-class and multi-label case, this is the average of the RS score of each class with weighting depending on the average parameter.

+ Best possible score is 1.0, higher value is better. Range = [0, 1]
+ https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
+ https://www.debadityachakravorty.com/ai-ml/cmatrix/
+ https://neptune.ai/blog/evaluation-metrics-binary-classification
+ https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score


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

	print(cm.recall_score(average=None))
	print(cm.RS(average="micro"))
	print(cm.RS(average="macro"))
	print(cm.RS(average="weighted"))

