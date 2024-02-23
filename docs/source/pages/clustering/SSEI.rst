Sum of Squared Error Index (SSEI)
=================================

.. toctree::
   :maxdepth: 3
   :caption: Sum of Squared Error Index (SSEI)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

Sum of Squared Error (SSE) is a commonly used metric to evaluate the quality of clustering in unsupervised learning problems.
SSE measures the sum of squared distances between each data point and its corresponding centroid or cluster center.
It quantifies the compactness of the clusters.

Here's how you can calculate the SSE in a clustering problem::

	1) Assign each data point to its nearest centroid or cluster center based on some distance metric (e.g., Euclidean distance).
	2) For each data point, calculate the squared Euclidean distance between the data point and its assigned centroid.
	3) Sum up the squared distances for all data points to obtain the SSE.

Higher SSE values indicate higher dispersion or greater variance within the clusters, while lower SSE values indicate
more compact and well-separated clusters. Therefore, minimizing the SSE is often a goal in clustering algorithms.

Example:

.. code-block:: python

	import numpy as np
	from permetrics import ClusteringMetric

	## For integer labels or categorical labels
	data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
	y_pred = np.array([0, 0, 1, 1, 1])

	cm = ClusteringMetric(X=data, y_pred=y_pred)

	print(cm.sum_squared_error_index())
	print(cm.SSEI())
