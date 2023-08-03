F2 Score (F2S)
==============

.. toctree::
   :maxdepth: 3
   :caption: F2 Score (F2S)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

It’s a metric that combines precision and recall, putting 2x emphasis on recall

.. math::

	F2 = 5 * (precision * recall) / (4 * precision + recall)

In the multi-class and multi-label case, this is the average of the F2 score of each class with weighting depending on the average parameter.

+ Best possible score is 1.0, higher value is better. Range = [0, 1]
+ https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
+ https://www.debadityachakravorty.com/ai-ml/cmatrix/
+ https://neptune.ai/blog/evaluation-metrics-binary-classification

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

	print(cm.f2_score(average=None))
	print(cm.F2S(average="micro"))
	print(cm.F2S(average="macro"))
	print(cm.F2S(average="weighted"))
