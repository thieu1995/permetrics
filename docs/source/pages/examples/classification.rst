Classification Metrics
======================

.. toctree::
   :maxdepth: 3
   :caption: Classification Metrics

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


Functional Style
----------------

+ This is a traditional way to call a specific metric you want to use. Everytime you want to use a metric, you need to pass y_true and y_pred

.. code-block:: python
	:emphasize-lines: 6,11,14-16,19-20

	## 1. Import packages, classes
	## 2. Create object
	## 3. From object call function and use

	import numpy as np
	from permetrics import ClassificationMetric

	y_true = [0, 1, 0, 0, 1, 0]
	y_pred = [0, 1, 0, 0, 0, 1]

	evaluator = ClassificationMetric()

	## 3.1 Call specific function inside object, each function has 2 names like below
	ps1 = evaluator.precision_score(y_true, y_pred)
	ps2 = evaluator.PS(y_true, y_pred)
	ps3 = evaluator.PS(y_true, y_pred)
	print(f"Precision: {ps1}, {ps2}, {ps3}")

	recall = evaluator.recall_score(y_true, y_pred)
	accuracy = evaluator.accuracy_score(y_true, y_pred)
	print(f"recall: {recall}, accuracy: {accuracy}")


Object-Oriented Style
---------------------

+ This is modern and better way to use metrics. You only need to pass y_true, y_pred one time when creating metric object.
+ After that, you can get the value of any metrics without passing y_true, y_pred

.. code-block:: python
	:emphasize-lines: 2,7,11-13

	import numpy as np
	from permetrics import ClassificationMetric

	y_true = [0, 1, 0, 0, 1, 0]
	y_pred = [0, 1, 0, 0, 0, 1]

	evaluator = ClassificationMetric(y_true, y_pred)

	## Get the result of any function you want to

	hamming_score = evaluator.hamming_score()
	mcc = evaluator.matthews_correlation_coefficient()
	specificity = evaluator.specificity_score()
	print(f"HL: {hamming_score}, MCC: {mcc}, specificity: {specificity}")
