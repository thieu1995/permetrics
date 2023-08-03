Duda Hart Index (DHI)
=====================

.. toctree::
   :maxdepth: 3
   :caption: Duda Hart Index (DHI)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

The Duda index, also known as the D-index or Duda-Hart index, is a clustering evaluation metric that measures the compactness and
separation of clusters. It was proposed by Richard O. Duda and Peter E. Hart in their book "Pattern Classification and Scene Analysis."

The Duda index is defined as the ratio between the average pairwise distance within clusters and the average pairwise distance
between clusters. A lower value of the Duda index indicates better clustering, indicating that the clusters are more compact
and well-separated. Here's the formula to calculate the Duda index::

	Duda Index = (Average pairwise intra-cluster distance) / (Average pairwise inter-cluster distance)

To calculate the Duda index, you need the following steps::

	Compute the average pairwise distance within each cluster (intra-cluster distance).
	Compute the average pairwise distance between different clusters (inter-cluster distance).
	Divide the average intra-cluster distance by the average inter-cluster distance to obtain the Duda index.

The Duda index is a useful metric for evaluating clustering results, particularly when the compactness and separation of
clusters are important. However, it's worth noting that the Duda index assumes Euclidean distance and may not work well
with all types of data or distance metrics.

When implementing the Duda index, you'll need to calculate the pairwise distances between data points within and between
clusters. You can use distance functions like Euclidean distance or other suitable distance metrics based on your
specific problem and data characteristics.


Example:

.. code-block:: python

	import numpy as np
	from permetrics import ClusteringMetric

	## For integer labels or categorical labels
	data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
	y_pred = np.array([0, 0, 1, 1, 1])

	cm = ClusteringMetric(X=data, y_pred=y_pred, decimal = 5)

	print(cm.duda_hart_index())
	print(cm.DHI())
