Beale Index (BI)
================

.. toctree::
   :maxdepth: 3
   :caption: Beale Index (BI)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

The Beale Index is a clustering validation metric that measures the quality of a clustering solution by computing the
ratio of the within-cluster sum of squares to the between-cluster sum of squares.
It is also known as the "variance ratio criterion" or the "F-ratio".

The within-cluster sum of squares is a measure of the variability of the data points within each cluster,
while the between-cluster sum of squares is a measure of the variability between the clusters.
The idea is that a good clustering solution should have low within-cluster variation and high
between-cluster variation, which results in a high Beale Index value.

The Beale Index can be calculated using the following formula::

	Beale Index = (sum of squared errors within clusters / degrees of freedom within clusters) / (sum of squared errors between clusters / degrees of freedom between clusters)

where the degrees of freedom are the number of data points minus the number of clusters, and the sum of squared errors is
the sum of the squared distances between each data point and the centroid of its assigned cluster.

The Beale Index ranges from 0 to infinity, with higher values indicating better clustering solutions.
However, the Beale Index has a tendency to favor solutions with more clusters, so it's important to
consider other metrics in conjunction with it.


Example:

.. code-block:: python

	import numpy as np
	from permetrics import ClusteringMetric

	## For integer labels or categorical labels
	data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
	y_pred = np.array([0, 0, 1, 1, 1])

	cm = ClusteringMetric(X=data, y_pred=y_pred, decimal = 5)

	print(cm.beale_index())
	print(cm.BI())
