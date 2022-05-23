Multiple Metrics
================

.. toctree::
   :maxdepth: 3
   :caption: Multiple Metrics

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


+ To reduce coding time when using multiple metrics. There are few ways to do it with permetrics by using OOP style

.. code-block:: python
	:emphasize-lines: 15,20,31

	import numpy as np
	from permetrics.regression import RegressionMetric

	y_true = np.array([3, -0.5, 2, 7, 5, 6])
	y_pred = np.array([2.5, 0.0, 2, 8, 5, 6])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)

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
	    "RMSE": {"decimal": 5},
	    "MAE": {"decimal": 4},
	    "MAPE": None,
	    "NSE": {"decimal": 3},
	}
	dict_result_3 = evaluator.get_metrics_by_dict(dict_metrics)
	print(dict_result_3)


.. code-block:: python
	:emphasize-lines: 2,7,10,13,17,21-27

	import numpy as np
	from permetrics.classification import ClassificationMetric

	y_true = [0, 1, 0, 0, 1, 0]
	y_pred = [0, 1, 0, 0, 0, 1]

	evaluator = ClassificationMetric(y_true, y_pred, decimal=5)

	## 1. Get list metrics by using loop
	list_metrics = ["PS", "RS", "LS", "SS"]
	list_results = []
	for metric in list_metrics:
	    list_results.append( evaluator.get_metric_by_name(metric) )
	print(list_results)

	## 2. Get list metrics by using function
	dict_result_2 = evaluator.get_metrics_by_list_names(list_metrics)
	print(dict_result_2)

	## 3. Get list metrics by using function and parameters
	dict_metrics = {
	    "PS": {"average": "micro"},
	    "RS": {"average": "macro"},
	    "LS": None,
	    "SS": {"average": "weighted"},
	}
	dict_result_3 = evaluator.get_metrics_by_dict(dict_metrics)
	print(dict_result_3)

