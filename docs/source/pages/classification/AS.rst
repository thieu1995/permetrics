Accuracy Score (AS)
===================

.. toctree::
   :maxdepth: 3
   :caption: Accuracy Score (AS)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. image:: /_static/images/class_score_1.png

In the multi-class and multi-label case, the "average of the AS score" refers to the average of the Accuracy Score (AS) for each class.
The weighting of the average depends on the average parameter, which determines the type of averaging to be performed.
There are several options for the average parameter, including "macro", "micro", and "weighted". Here's a description of each averaging method:
* Macro averaging: Calculates the AS for each class independently and then takes the average. Each class is given equal weight, regardless of its size or
distribution. This averaging method treats all classes equally.
* Micro averaging: Calculates the AS by considering the total number of true positives, false negatives, and false positives across all classes. This method
gives more weight to classes with larger numbers of instances.
* Weighted averaging: Similar to macro averaging, this method calculates the AS for each class independently and takes the average. However, each class is
given a weight proportional to its number of instances. This means that classes with more instances contribute more to the overall average.

The choice of averaging method depends on the specific requirements and characteristics of the problem at hand. It's important to consider the class
distribution, class imbalance, and the desired focus of evaluation when selecting the appropriate averaging method.

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

	print(cm.accuracy_score(average=None))
	print(cm.accuracy_score(average="micro"))
	print(cm.AS(average="macro"))
	print(cm.AS(average="weighted"))

