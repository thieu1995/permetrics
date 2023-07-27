Multiple Outputs Multiple Metrics
=================================

.. toctree::
   :maxdepth: 3
   :caption: Multiple Outputs Multiple Metrics

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


+ Scikit-learn library is limited with multi-output metrics, but permetrics can produce multi-output for all of metrics

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

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)

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
	    {"decimal": 3, "multi_output": "mean"},
	    {"decimal": 4, "multi_output": [0.5, 0.2, 0.1, 0.2]},
	    {"decimal": 5, "multi_output": "raw_values"}
	]
	dict_result_1 = evaluator.get_metrics_by_list_names(list_metrics, list_paras)
	print(dict_result_1)


	## 2. Get list metrics by using dict_metrics
	dict_metrics = {
	    "RMSE": {"decimal": 5, "multi_output": "mean"},
	    "MAE": {"decimal": 4, "multi_output": "raw_values"},
	    "MSE": {"decimal": 2, "multi_output": [0.5, 0.2, 0.1, 0.2]},
	}
	dict_result_2 = evaluator.get_metrics_by_dict(dict_metrics)
	print(dict_result_2)
