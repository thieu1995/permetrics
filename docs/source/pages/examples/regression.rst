Regression Metrics
==================

.. toctree::
   :maxdepth: 3
   :caption: Regression Metrics

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
	mae = evaluator.MAE(y_true, y_pred)
	print(f"MSE: {mse}, MAE: {mae}")


Object-Oriented Style
---------------------

+ This is modern and better way to use metrics. You only need to pass y_true, y_pred one time when creating metric object.
+ After that, you can get the value of any metrics without passing y_true, y_pred

.. code-block:: python
	:emphasize-lines: 11,14-16

	## 1. Import packages, classes
	## 2. Create object
	## 3. From object call function and use

	import numpy as np
	from permetrics import RegressionMetric

	y_true = np.array([3, -0.5, 2, 7, 5, 6])
	y_pred = np.array([2.5, 0.0, 2, 8, 5, 6])

	evaluator = RegressionMetric(y_true, y_pred)

	## Get the result of any function you want to
	rmse = evaluator.RMSE()
	mse = evaluator.MSE()
	mae = evaluator.MAE()

	print(f"RMSE: {rmse}, MSE: {mse}, MAE: {mae}")


Multiple Metrics Style
----------------------

+ To reduce coding time when using multiple metrics. There are few ways to do it with Permetrics by using `OOP style`

.. code-block:: python
	:emphasize-lines: 15,20,31

	import numpy as np
	from permetrics import RegressionMetric

	y_true = np.array([3, -0.5, 2, 7, 5, 6])
	y_pred = np.array([2.5, 0.0, 2, 8, 5, 6])

	evaluator = RegressionMetric(y_true, y_pred)

	## Define list of metrics you want to use
	list_metrics = ["RMSE", "MAE", "MAPE", "NSE"]

	## 1. Get list metrics by using loop
	list_results = []
	for metric in list_metrics:
		list_results.append( evaluator.get_metric_by_name(metric) )
	print(list_results)


	## 2. Get list metrics by using function
	dict_result_2 = evaluator.get_metrics_by_list_names(list_metrics)
	print(dict_result_2)


	## 3. Get list metrics by using function and parameters
	dict_metrics = {
	    "RMSE": None,
	    "MAE": None,
	    "MAPE": None,
	    "NSE": None,
	}
	dict_result_3 = evaluator.get_metrics_by_dict(dict_metrics)
	print(dict_result_3)


Multiple Outputs for Multiple Metrics
-------------------------------------

+ The Scikit-learn library is limited with multi-output metrics, but Permetrics can produce multi-output for all of metrics

.. code-block:: python

	import numpy as np
	from permetrics import RegressionMetric

	## This y_true and y_pred have 4 columns, 4 outputs
	y_true = np.array([ [3, -0.5, 2, 7],
	                    [5, 6, -0.3, 9],
	                    [-11, 23, 8, 3.9] ])

	y_pred = np.array([ [2.5, 0.0, 2, 8],
	                    [5.2, 5.4, 0, 9.1],
	                    [-10, 23, 8.2, 4] ])

	evaluator = RegressionMetric(y_true, y_pred)

	## 1. By default, all metrics can automatically return the multi-output results
	# rmse = evaluator.RMSE()
	# print(rmse)

	## 2. If you want to take mean of all outputs, can set the parameter: multi-output = "mean"
	# rmse_2 = evaluator.RMSE(multi_output="mean")
	# print(rmse_2)

	## 3. If you want a specific metric has more important than other, you can set weight for each output.
	# rmse_3 = evaluator.RMSE(multi_output=[0.5, 0.05, 0.1, 0.35])
	# print(rmse_3)


	## Get multiple metrics with multi-output or single-output by parameters


	## 1. Get list metrics by using list_names
	list_metrics = ["RMSE", "MAE", "MSE"]
	list_paras = [
	    {"multi_output": "mean"},
	    {"multi_output": [0.5, 0.2, 0.1, 0.2]},
	    {"multi_output": "raw_values"}
	]
	dict_result_1 = evaluator.get_metrics_by_list_names(list_metrics, list_paras)
	print(dict_result_1)


	## 2. Get list metrics by using dict_metrics
	dict_metrics = {
	    "RMSE": {"multi_output": "mean"},
	    "MAE": {"multi_output": "raw_values"},
	    "MSE": {"multi_output": [0.5, 0.2, 0.1, 0.2]},
	}
	dict_result_2 = evaluator.get_metrics_by_dict(dict_metrics)
	print(dict_result_2)
