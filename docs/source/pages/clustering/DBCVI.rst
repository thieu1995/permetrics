Density-Based Clustering Validation Index (DBCVI)
=================================================

.. toctree::
   :maxdepth: 3
   :caption: Density-Based Clustering Validation Index (DBCVI)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

The Density-Based Clustering Validation (DBCV) metric is another clustering validation metric that is used to evaluate the quality
of a clustering solution, particularly for density-based clustering algorithms such as DBSCAN.

The DBCV metric measures the average ratio of the distances between the data points and their cluster centroids, to the
distances between the data points and the nearest data points in other clusters. The idea is that a good clustering solution
should have compact and well-separated clusters, so the ratio of these distances should be high.

The DBCV metric is calculated using the following formula::

	DBCV = (1 / n) * sum_{i=1}^n (sum_{j=1}^n (d(i,j) / max{d(i,k), k!=j}))

where n is the number of data points, d(i,j) is the Euclidean distance between data points i and j, and max{d(i,k), k!=j} is
the maximum distance between data point i and any other data point in a different cluster.

The DBCV metric ranges from 0 to 1, with lower values indicating better clustering solutions. A value of 0 indicates a perfect
clustering solution, where all data points belong to their own cluster and the distances between clusters are maximized.


Example:

.. code-block:: python

	import numpy as np
	from permetrics import ClusteringMetric

	## For integer labels or categorical labels
	data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
	y_pred = np.array([0, 0, 1, 1, 1])

	cm = ClusteringMetric(X=data, y_pred=y_pred)

	print(cm.density_based_clustering_validation_index())
	print(cm.DBCVI())
