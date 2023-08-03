Entropy Score (ES)
==================

.. toctree::
   :maxdepth: 3
   :caption: Entropy Score (ES)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

Entropy is a metric used to evaluate the quality of clustering results, particularly when the ground truth labels of the data
points are known. It measures the amount of uncertainty or disorder within the clusters produced by a clustering algorithm.

Here's how the Entropy score is calculated::

	1) For each cluster, compute the class distribution by counting the occurrences of each class label within the cluster.
	2) Normalize the class distribution by dividing the count of each class label by the total number of data points in the cluster.
	3) Compute the entropy for each cluster using the normalized class distribution.
	4) Weight the entropy of each cluster by its relative size (proportion of data points in the whole dataset).
	5) Sum up the weighted entropies of all clusters.

The resulting value is the Entropy score, which typically ranges from 0 to 1. A lower Entropy score indicates better clustering,
as it implies more purity and less uncertainty within the clusters.

Entropy score considers both the composition of each cluster and the distribution of classes within the clusters.
It provides a more comprehensive evaluation of clustering performance compared to simple metrics like Purity.


Example:

.. code-block:: python

	import numpy as np
	from permetrics import ClusteringMetric

	## For integer labels or categorical labels
	y_true = np.array([0, 0, 1, 1, 1, 2, 2, 1])
	y_pred = np.array([0, 0, 1, 1, 2, 2, 2, 2])

	cm = ClusteringMetric(y_true=y_true, y_pred=y_pred, decimal = 5)

	print(cm.entropy_score())
	print(cm.ES())
