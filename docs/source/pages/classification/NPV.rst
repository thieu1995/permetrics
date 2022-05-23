Negative Predictive Value (NPV)
===============================

.. toctree::
   :maxdepth: 3
   :caption: Negative Predictive Value (NPV)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. image:: /_static/images/class_score_1.png

The negative predictive value is defined as the number of true negatives (people who test negative who don't have a condition) divided
by the total number of people who test negative.

The negative predictive value is the ratio tn / (tn + fn) where tn is the number of true negatives and fn the number of false negatives.

In the multi-class and multi-label case, this is the average of the NPV score of each class with weighting depending on the average parameter.

+ Best possible score is 1.0, higher value is better. Range = [0, 1]
+ https://www.debadityachakravorty.com/ai-ml/cmatrix/
+ https://neptune.ai/blog/evaluation-metrics-binary-classification
+ https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2



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

	print(cm.npv())
	print(cm.NPV(average="micro"))
	print(cm.NPV(average="macro"))
	print(cm.NPV(average="weighted"))

