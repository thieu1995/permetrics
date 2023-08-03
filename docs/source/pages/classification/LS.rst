Lift Score (LS)
===============

.. toctree::
   :maxdepth: 3
   :caption: Lift Score (LS)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


In the multi-class and multi-label case, this is the average of the LS score of each class with weighting depending on the average parameter.

+ Higher is better (No best value), Range = [0, +inf)
+ http://rasbt.github.io/mlxtend/user_guide/evaluate/lift_score/
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

	print(cm.lift_score(average=None))
	print(cm.LS(average="micro"))
	print(cm.LS(average="macro"))
	print(cm.LS(average="weighted"))

