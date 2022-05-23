OOP Style
=========

.. toctree::
   :maxdepth: 3
   :caption: OOP Style

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


+ This is modern and better way to use metrics. You only need to pass y_true, y_pred one time when creating metric object.
+ After that, you can get the value of any metrics without passing y_true, y_pred

.. code-block:: python
	:emphasize-lines: 11,14-16

	## 1. Import packages, classes
	## 2. Create object
	## 3. From object call function and use

	import numpy as np
	from permetrics.regression import RegressionMetric

	y_true = np.array([3, -0.5, 2, 7, 5, 6])
	y_pred = np.array([2.5, 0.0, 2, 8, 5, 6])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)

	## Get the result of any function you want to
	rmse = evaluator.RMSE()
	mse = evaluator.MSE()
	mae = evaluator.MAE()

	print(f"RMSE: {rmse}, MSE: {mse}, MAE: {mae}")


.. code-block:: python
	:emphasize-lines: 2,7,11-13

	import numpy as np
	from permetrics.classification import ClassificationMetric

	y_true = [0, 1, 0, 0, 1, 0]
	y_pred = [0, 1, 0, 0, 0, 1]

	evaluator = ClassificationMetric(y_true, y_pred, decimal=5)

	## Get the result of any function you want to

	hamming_loss = evaluator.hamming_loss()
	mcc = evaluator.matthews_correlation_coefficient()
	specificity = evaluator.specificity_score()
	print(f"HL: {hamming_loss}, MCC: {mcc}, specificity: {specificity}")

