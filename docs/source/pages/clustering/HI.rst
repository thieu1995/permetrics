Hartigan Index (HI)
===================

.. toctree::
   :maxdepth: 3
   :caption: Hartigan Index (HI)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

The Hartigan index, also known as the Hartigan's criterion, is a measure used for evaluating the quality of clustering solutions.
It is specifically designed for assessing the goodness of fit of a clustering algorithm, particularly the k-means algorithm.

.. image:: /_static/images/HI.png

The Hartigan index quantifies the compactness of clusters and the separation between clusters in a clustering solution.
It aims to find a balance between minimizing the within-cluster variance (compactness) and maximizing the between-cluster variance (separation).

While the Hartigan index is a useful measure, it is not as widely used as other clustering evaluation indices
like the Silhouette coefficient or Dunn index. Nevertheless, it can provide insights into the quality of a clustering solution,
particularly when comparing different clustering algorithms or determining the optimal number of clusters.

The goal of the Hartigan index is to minimize this ratio. Lower values of the Hartigan index indicate better clustering
solutions with lower within-cluster variance and higher separation between clusters.


Example:

.. code-block:: python

	import numpy as np
	from permetrics import ClusteringMetric

	## For integer labels or categorical labels
	data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
	y_pred = np.array([0, 0, 1, 1, 1])

	cm = ClusteringMetric(X=data, y_pred=y_pred, decimal = 5)

	print(cm.hartigan_index())
	print(cm.HI())
