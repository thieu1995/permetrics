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

The Hartigan index quantifies the compactness of clusters and the separation between clusters in a clustering solution.
It aims to find a balance between minimizing the within-cluster variance (compactness) and maximizing the between-cluster variance (separation).

The Hartigan index is calculated by comparing the within-cluster sum of squares (WCSS) of a clustering solution
with the expected WCSS under a null hypothesis of random clustering. The index is defined as the ratio of the
observed WCSS to the expected WCSS. The goal is to minimize this ratio, and lower values of the Hartigan index
indicate better clustering solutions.

It's worth noting that the Hartigan index is primarily used as an internal evaluation measure, meaning it
assesses the quality of a clustering solution based on the characteristics of the data and the algorithm itself.
It does not require external information or ground truth labels for evaluation.

While the Hartigan index is a useful measure, it is not as widely used as other clustering evaluation indices
like the Silhouette coefficient or Dunn index. Nevertheless, it can provide insights into the quality of a clustering solution,
particularly when comparing different clustering algorithms or determining the optimal number of clusters.

The Hartigan index is calculated using the within-cluster sum of squares (WCSS) of a clustering solution and the expected WCSS
under a null hypothesis of random clustering. Here are the equations for calculating the Hartigan index.

1. Calculate the within-cluster sum of squares (WCSS) for the clustering solution::

   Let C be the number of clusters.
   Let n be the total number of data points.
   Let X_i be the data point i.
   Let μ_k be the centroid of cluster k.
   WCSS = Σ_i=1 to n Σ_k=1 to C (X_i - μ_k)^2

2. Calculate the expected WCSS under the null hypothesis of random clustering::

   Assuming random clustering, the centroids are calculated by randomly assigning data points to clusters and calculating the WCSS.
   Repeat this process multiple times (typically 10 or more) to get the expected WCSS.

3. Calculate the Hartigan index::

   Hartigan index = (WCSS) / (Expected WCSS)


The goal of the Hartigan index is to minimize this ratio. Lower values of the Hartigan index indicate better clustering
solutions with lower within-cluster variance and higher separation between clusters.

It's important to note that the specific implementation of the Hartigan index may vary slightly depending on the clustering
algorithm being used and any modifications made to the original formulation. However, the core idea of comparing the
observed WCSS with the expected WCSS remains the same.


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
