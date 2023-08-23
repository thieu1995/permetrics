Dunn Index (DI)
===============

.. toctree::
   :maxdepth: 3
   :caption: Dunn Index (DI)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


The Dunn Index, which is a measure used to evaluate the performance of clustering algorithms. The Dunn Index aims to quantify the compactness and separation
between clusters in a clustering solution. It helps assess the quality of the clustering by considering both the distance between points within the same cluster (intra-cluster distance) and the distance between points in different clusters (inter-cluster distance).


.. image:: /_static/images/DI.png


A higher Dunn Index value indicates better clustering quality – it suggests that the clusters are well separated from each other while being compact internally. Conversely, a lower Dunn Index value may indicate that the clusters are too spread out or not well separated.

However, like any clustering evaluation metric, the Dunn Index has its limitations and should be used in conjunction with other metrics and domain knowledge. It's worth noting that the choice of clustering algorithm, distance metric, and dataset characteristics can influence the interpretation of the Dunn Index.


Example:

.. code-block:: python

	import numpy as np
	from permetrics import ClusteringMetric

	## For integer labels or categorical labels
	data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
	y_pred = np.array([0, 0, 1, 1, 1])

	cm = ClusteringMetric(X=data, y_pred=y_pred, decimal = 5)

	print(cm.dunn_index())
	print(cm.DI())
