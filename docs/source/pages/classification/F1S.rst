F1 Score (F1S)
==============

.. toctree::
   :maxdepth: 3
   :caption: F1 Score (F1S)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. image:: /_static/images/class_score_1.png

Compute the F1 score, also known as balanced F-score or F-measure.

The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is

.. math::

	F1 = 2 * (precision * recall) / (precision + recall)

In the multi-class and multi-label case, this is the average of the F1 score of each class with weighting depending on the average parameter.

+ Best possible score is 1.0, higher value is better. Range = [0, 1]
+ https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
+ https://www.debadityachakravorty.com/ai-ml/cmatrix/
+ https://neptune.ai/blog/evaluation-metrics-binary-classification
+ https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html


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

	print(cm.f1_score(average=None))
	print(cm.F1S(average="micro"))
	print(cm.F1S(average="macro"))
	print(cm.F1S(average="weighted"))
