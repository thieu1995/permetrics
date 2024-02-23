Specificity Score (SS)
======================

.. toctree::
   :maxdepth: 3
   :caption: Specificity Score (SS)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. image:: /_static/images/class_score_1.png

The specificity score is the ratio tn / (tn + fp) where tn is the number of false positives and fp the number of false positives.
It measures how many observations out of all negative observations have we classified as negative.
In fraud detection example, it tells us how many transactions, out of all non-fraudulent transactions, we marked as clean.

In the multi-class and multi-label case, this is the average of the SS score of each class with weighting depending on the average parameter.


+ Best possible score is 1.0, higher value is better. Range = [0, 1]
+ https://neptune.ai/blog/evaluation-metrics-binary-classification
+ https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
+ https://www.debadityachakravorty.com/ai-ml/cmatrix/

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

	cm = ClassificationMetric(y_true, y_pred)

	print(cm.specificity_score(average=None))
	print(cm.ss(average="micro"))
	print(cm.SS(average="macro"))
	print(cm.SS(average="weighted"))
