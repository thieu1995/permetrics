Hamming Score (HS)
==================

.. toctree::
   :maxdepth: 3
   :caption: Hamming Score (HS)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


The Hamming score is 1 - the fraction of labels that are incorrectly predicted.

In the multi-class and multi-label case, this is the average of the HL score of each class with weighting depending on the average parameter.

+ Higher is better (Best = 1), Range = [0, 1]
+ A little bit difference than hamming_score in scikit-learn library.
+ https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_score.html#sklearn.metrics.hamming_score


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

	print(cm.hamming_score(average=None))
	print(cm.HS(average="micro"))
	print(cm.HS(average="macro"))
	print(cm.HS(average="weighted"))

