Hamming Loss (HL)
=================

.. toctree::
   :maxdepth: 3
   :caption: Hamming Loss (HL)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


The Hamming loss is the fraction of labels that are incorrectly predicted.

In the multi-class and multi-label case, this is the average of the HL score of each class with weighting depending on the average parameter.

+ Best possible score is 0.0, lower value is better. Range = [0, 1]
+ https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html#sklearn.metrics.hamming_loss


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

	print(cm.hamming_loss(average=None))
	print(cm.hl(average="micro"))
	print(cm.HL(average="macro"))
	print(cm.HL(average="weighted"))

