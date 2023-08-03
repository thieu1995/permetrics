PCD - Prediction of Change in Direction
=======================================

.. toctree::
   :maxdepth: 3
   :caption: PCD - Prediction of Change in Direction

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


.. math::

	\text{PCD}(y, \hat{y}) = \frac{1}{n-1} \sum_{i=2}^{n} I\left((f_{i}-f_{i-1}) (y_{i}-y_{i-1}) > 0\right)

+ where $f_i$ is the predicted value at time $i$, $y_i$ is the actual value at time $i$, $n$ is the total number of predictions, and $I(\cdot)$ is the indicator function which equals 1 if the argument is true and 0 otherwise.


+ Best possible score is 1.0, bigger value is better . Range = [0, 1]
+ The Prediction of Change in Direction (PCD) metric is used to evaluate the performance of regression models on detecting changes in the direction of a target variable.


Latex equation code::

	\text{PCD}(y, \hat{y}) = \frac{1}{n-1} \sum_{i=2}^{n} I\left((f_{i}-f_{i-1}) (y_{i}-y_{i-1}) > 0\right)


Example to use PCD metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.prediction_of_change_in_direction())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6]])
	y_pred = array([[0, 2], [-1, 2], [8, -5]])

	evaluator = RegressionMetric(y_true, y_pred, decimal=5)
	print(evaluator.PCD(multi_output="raw_values"))
