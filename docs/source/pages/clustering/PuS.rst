Purity Score (PuS)
==================

.. toctree::
   :maxdepth: 3
   :caption: Purity Score (PuS)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

Purity is a metric used to evaluate the quality of clustering results, particularly in situations where the ground truth
labels of the data points are known. It measures the extent to which the clusters produced by a clustering algorithm
match the true class labels of the data. Here's how Purity is calculated::

	1) For each cluster, find the majority class label among the data points in that cluster.
	2) Sum up the sizes of the clusters that belong to the majority class label.
	3) Divide the sum by the total number of data points.

The resulting value is the Purity score, which ranges from 0 to 1. A Purity score of 1 indicates a perfect clustering,
where each cluster contains only data points from a single class.

Purity is a simple and intuitive metric but has some limitations. It does not consider the actual structure or
distribution of the data within the clusters and is sensitive to the number of clusters and class imbalance.
Therefore, it may not be suitable for evaluating clustering algorithms in all scenarios.

Example:

.. code-block:: python

	import numpy as np
	from permetrics import ClusteringMetric

	## For integer labels or categorical labels
	y_true = np.array([0, 0, 1, 1, 1, 2, 2, 1])
	y_pred = np.array([0, 0, 1, 1, 2, 2, 2, 2])

	cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)

	print(cm.purity_score())
	print(cm.PuS())
