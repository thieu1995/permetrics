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
	from permetrics.regression import RegressionMetric

	y_true = np.array([3, -0.5, 2, 7, 5, 6])
	y_pred = np.array([2.5, 0.0, 2, 8, 5, 6])

	evaluator = RegressionMetric()

	## 3.1 Call specific function inside object, each function has 3 names like below
	rmse_1 = evaluator.RMSE(y_true, y_pred)
	rmse_2 = evaluator.rmse(y_true, y_pred)
	rmse_3 = evaluator.root_mean_squared_error(y_true, y_pred)
	print(f"RMSE: {rmse_1}, {rmse_2}, {rmse_3}")

	mse = evaluator.MSE(y_true, y_pred)
	mae = evaluator.MAE(y_true, y_pred, decimal=5)
	print(f"MSE: {mse}, MAE: {mae}")

