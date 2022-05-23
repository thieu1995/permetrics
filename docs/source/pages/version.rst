Version
=======


+ Permetrics version >= 1.2.0::

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.RMSE())
	print(evaluator.MSE())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.RMSE(multi_output="raw_values", decimal=5))
	print(evaluator.MAE(multi_output="raw_values", decimal=5))


	## All metrics

	EVS = evs = explained_variance_score
	ME = me = max_error
	MBE = mbe = mean_bias_error
	MAE = mae = mean_absolute_error
	MSE = mse = mean_squared_error
	RMSE = rmse = root_mean_squared_error
	MSLE = msle = mean_squared_log_error
	MedAE = medae = median_absolute_error
	MRE = mre = MRB = mrb = mean_relative_bias = mean_relative_error
	MPE = mpe = mean_percentage_error
	MAPE = mape = mean_absolute_percentage_error
	SMAPE = smape = symmetric_mean_absolute_percentage_error
	MAAPE = maape = mean_arctangent_absolute_percentage_error
	MASE = mase = mean_absolute_scaled_error
	NSE = nse = nash_sutcliffe_efficiency
	NNSE = nnse = normalized_nash_sutcliffe_efficiency
	WI = wi = willmott_index
	R = r = PCC = pcc = pearson_correlation_coefficient
	AR = ar = APCC = apcc = absolute_pearson_correlation_coefficient
	R2s = r2s = pearson_correlation_coefficient_square
	CI = ci = confidence_index
	COD = cod = R2 = r2 = coefficient_of_determination
	ACOD = acod = AR2 = ar2 = adjusted_coefficient_of_determination
	DRV = drv = deviation_of_runoff_volume
	KGE = kge = kling_gupta_efficiency
	GINI = gini = gini_coefficient
	GINI_WIKI = gini_wiki = gini_coefficient_wiki
	PCD = pcd = prediction_of_change_in_direction
	CE = ce = cross_entropy
	KLD = kld = kullback_leibler_divergence
	JSD = jsd = jensen_shannon_divergence
	VAF = vaf = variance_accounted_for
	RAE = rae = relative_absolute_error
	A10 = a10 = a10_index
	A20 = a20 = a20_index
	A30 = a30 = a30_index
	NRMSE = nrmse = normalized_root_mean_square_error
	RSE = rse = residual_standard_error

	RE = re = RB = rb = single_relative_bias = single_relative_error
	AE = ae = single_absolute_error
	SE = se = single_squared_error
	SLE = sle = single_squared_log_error



+ Permetrics version <= 1.1.3::

	##  All you need to do is: (Make sure your y_true and y_pred is a numpy array)
	## For example with RMSE:

	from numpy import array
	from permetrics.regression import Metrics

	## 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	y_true2 = array([3, -0.5, 2, 7])
	y_pred2 = array([2.5, 0.0, 2, 9])

	### C1. Using OOP style - very powerful when calculating multiple metrics
	obj1 = Metrics(y_true, y_pred)  # Pass the data here
	result = obj1.root_mean_squared_error(clean=True, decimal=5)
	print(f"1-D array, OOP style: {result}")

	### C2. Using functional style
	obj2 = Metrics()
	result = obj2.root_mean_squared_error(clean=True, decimal=5, y_true=y_true2, y_pred=y_pred2)
	# Pass the data here, remember the keywords (y_true, y_pred)
	print(f"1-D array, Functional style: {result}")

	## > 1-D array - Multi-dimensional Array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	multi_outputs = [None, "raw_values", [0.3, 1.2], array([0.5, 0.2]), (0.1, 0.9)]
	obj3 = Metrics(y_true, y_pred)
	for multi_output in multi_outputs:
	    result = obj3.root_mean_squared_error(clean=False, multi_output=multi_output, decimal=5)
	    print(f"n-D array, OOP style: {result}")

	# Or run the simple:
	python examples/RMSE.py



# The more complicated tests in the folder: examples
```




.. toctree::
   :maxdepth: 3
   :caption: Example for different version

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3
