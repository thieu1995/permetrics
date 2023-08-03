Matthews Correlation Coefficient (MCC)
======================================

.. toctree::
   :maxdepth: 3
   :caption: Matthews Correlation Coefficient (MCC)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


In the multi-class and multi-label case, this is the average of the MCC score of each class with weighting depending on the average parameter.

+ Best possible score is 1.0, higher value is better. Range = [-1, +1]
+ https://neptune.ai/blog/evaluation-metrics-binary-classification
+ https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef


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

	print(cm.mcc(average=None))
	print(cm.MCC(average="micro"))
	print(cm.MCC(average="macro"))
	print(cm.MCC(average="weighted"))

