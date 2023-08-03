Functional Style
================

.. toctree::
   :maxdepth: 3
   :caption: Functional Style

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3



+ This is traditional way to call a specific metric you want to use. Everytime you want to use a function, you need to pass y_true and y_pred

.. code-block:: python
	:emphasize-lines: 11,14-16

	## 1. Import packages, classes
	## 2. Create object
	## 3. From object call function and use

	import numpy as np
	from permetrics import RegressionMetric

	y_true = np.array([3, -0.5, 2, 7, 5, 6])
	y_pred = np.array([2.5, 0.0, 2, 8, 5, 6])

	evaluator = RegressionMetric()

	## 3.1 Call specific function inside object, each function has 2 names like below
	rmse_1 = evaluator.RMSE(y_true, y_pred)
	rmse_2 = evaluator.root_mean_squared_error(y_true, y_pred)
	print(f"RMSE: {rmse_1}, {rmse_2}")

	mse = evaluator.MSE(y_true, y_pred)
	mae = evaluator.MAE(y_true, y_pred, decimal=5)
	print(f"MSE: {mse}, MAE: {mae}")



.. code-block:: python
	:emphasize-lines: 2,7,9-11,14-15

	import numpy as np
	from permetrics import ClassificationMetric

	y_true = [0, 1, 0, 0, 1, 0]
	y_pred = [0, 1, 0, 0, 0, 1]

	evaluator = ClassificationMetric()

	ps1 = evaluator.precision_score(y_true, y_pred, decimal=5)
	ps2 = evaluator.PS(y_true, y_pred, decimal=3)
	ps3 = evaluator.PS(y_true, y_pred, decimal=4)
	print(f"Precision: {ps1}, {ps2}, {ps3}")

	recall = evaluator.recall_score(y_true, y_pred)
	accuracy = evaluator.accuracy_score(y_true, y_pred)
	print(f"recall: {recall}, accuracy: {accuracy}")


.. code-block:: python
	:emphasize-lines: 2,7,9-11,14-15

	import numpy as np
	from permetrics import ClusteringMetric

	y_true = [0, 1, 0, 0, 1, 0]
	y_pred = [0, 1, 0, 0, 0, 1]

	evaluator = ClusteringMetric()

	ps1 = evaluator.mutual_info_score(y_true, y_pred, decimal=5)
	ps2 = evaluator.MIS(y_true, y_pred, decimal=3)
	print(f"Mutual Information score: {ps1}, {ps2}")

	homogeneity = evaluator.homogeneity_score(y_true, y_pred)
	completeness  = evaluator.CS(y_true, y_pred)
	print(f"Homogeneity: {homogeneity}, Completeness : {completeness}")
