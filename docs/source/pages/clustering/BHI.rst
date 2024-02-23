Ball Hall Index
===============

.. toctree::
   :maxdepth: 3
   :caption: Ball Hall Index

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

The Ball Hall Index is a clustering validity index that measures the compactness and separation of clusters in a clustering result. It provides a quantitative measure of how well-separated and tight the clusters are.

The formula for calculating the Ball Hall Index is as follows::

   BHI = Xichma(1 / (2 * n_i) * Xichma(d(x, c_i)) / n

Where:

n is the total number of data points
n_i is the number of data points in cluster i
d(x, c_i) is the Euclidean distance between a data point x and the centroid c_i of cluster i

The Ball Hall Index computes the average distance between each data point and its cluster centroid and then averages this across all clusters. The index is inversely proportional to the compactness and separation of the clusters. A smaller BHI value indicates better-defined and well-separated clusters.

A lower BHI value indicates better clustering, as it signifies that the data points are closer to their own cluster centroid than to the centroids of other clusters, indicating a clear separation between clusters.

The Ball Hall Index is often used as an internal evaluation metric for clustering algorithms to compare different clustering results or to determine the optimal number of clusters. However, it should be noted that it is not without limitations and should be used in conjunction with other evaluation metrics and domain knowledge for a comprehensive assessment of clustering results.


Example:

.. code-block:: python

	import numpy as np
	from permetrics import ClusteringMetric

	## For integer labels or categorical labels
	data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
	y_pred = np.array([0, 0, 1, 1, 1])

	cm = ClusteringMetric(X=data, y_pred=y_pred)

	print(cm.ball_hall_index())
	print(cm.BHI())
