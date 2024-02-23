Calinski-Harabasz Index
=======================

.. toctree::
   :maxdepth: 3
   :caption: Calinski-Harabasz Index

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

The Calinski-Harabasz Index is a clustering evaluation metric used to measure the quality of clusters obtained from clustering algorithms. It aims to quantify the separation between clusters and the compactness within clusters.

.. image:: /_static/images/CHI.png

In practice, you can use the Calinski-Harabasz Index along with other clustering evaluation metrics to assess the performance of clustering algorithms and select the best number of clusters for your dataset.


Example:

.. code-block:: python

	import numpy as np
	from permetrics import ClusteringMetric

	## For integer labels or categorical labels
	data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
	y_pred = np.array([0, 0, 1, 1, 1])

	cm = ClusteringMetric(X=data, y_pred=y_pred)

	print(cm.calinski_harabasz_index())
	print(cm.CHI())
