R-Squared Index (RSI)
=====================

.. toctree::
   :maxdepth: 3
   :caption: R-Squared Index (RSI)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

The R-squared index is another clustering validation metric that is used to measure the quality of a clustering solution.
It is based on the idea of comparing the variance of the data before and after clustering.
The R-squared index measures the proportion of the total variance in the data that is explained by the clustering solution.

The R-squared index is calculated using the following formula::

	R-squared = (total variance - variance within clusters) / total variance

where total variance is the variance of the entire dataset, and variance within clusters is the sum of the variances
of each cluster.
The R-squared index ranges from -inf to 1, with higher values indicating better clustering solutions. A negative value indicates that the clustering
solution is worse than random, while a value of 0 indicates that the clustering solution explains no variance beyond chance.
A value of 1 indicates that the clustering solution perfectly explains all the variance in the data.


Note that the R-squared index has some limitations, as it can be biased towards solutions with more clusters.
It is also sensitive to the scale and dimensionality of the data, and may not be appropriate for all clustering problems.
Therefore, it's important to consider multiple validation metrics when evaluating clustering solutions.

Example:

.. code-block:: python

	import numpy as np
	from permetrics import ClusteringMetric

	## For integer labels or categorical labels
	data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
	y_pred = np.array([0, 0, 1, 1, 1])

	cm = ClusteringMetric(X=data, y_pred=y_pred, decimal = 5)

	print(cm.r_squared_index())
	print(cm.RSI())
