Clustering Metrics
==================

.. toctree::
   :maxdepth: 3
   :caption: Clustering Metrics

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


Note that, this type of metrics is kinda differ from regression and classification. There are two type of clustering metrics include internal and external
metrics, each serving a different purpose:

1. Internal Metrics:

	* Objective: Internal metrics evaluate the quality of clusters based on the data itself without relying on external information or ground truth labels.
	* Example Metrics:

		* Silhouette Score: Measures how well-separated clusters are.
		* Davies-Bouldin Score: Computes the compactness and separation of clusters.
		* Inertia (within-cluster sum of squares): Measures how far points within a cluster are from the cluster's centroid.

2. External Metrics:

	* Objective: External metrics assess the quality of clusters by comparing them to some external criterion, often ground truth labels or known groupings.
	* Example Metrics:

		* Adjusted Rand Index (ARI): Measures the similarity between true and predicted clusters, adjusted for chance.
		* Normalized Mutual Information Index (NMII): Measures the mutual information between true and predicted clusters, normalized.
		* Fowlkes-Mallows Index: Computes the geometric mean of precision and recall between true and predicted clusters.

While internal metrics provide insights into the structure of the data within the clusters, external metrics help evaluate clustering performance against a
known or expected structure, such as labeled data. The choice between internal and external metrics depends on the availability of ground truth information
and the specific goals of the clustering analysis.

To clearly distinguish between internal and external clustering metrics, we use specific suffixes in their function names. Using the suffix `index` can
indicate internal clustering metrics, while using the suffix `score` can indicate external clustering metrics. This naming convention makes it easier for
users to differentiate between the two types of metrics and facilitates their usage.

By following this convention, users can easily identify whether a metric is designed for evaluating the quality of clusters within a dataset (internal) or
for comparing clusters to external reference labels or ground truth (external). This distinction is important because internal and external metrics serve
different purposes and have different interpretations.


Functional Style
----------------

+ External clustering metrics

.. code-block:: python
	:emphasize-lines: 2,7,9,10,13,14

	import numpy as np
	from permetrics import ClusteringMetric

	y_true = [0, 1, 0, 0, 1, 0]
	y_pred = [0, 1, 0, 0, 0, 1]

	evaluator = ClusteringMetric()

	ps1 = evaluator.mutual_info_score(y_true, y_pred)
	ps2 = evaluator.MIS(y_true, y_pred)
	print(f"Mutual Information score: {ps1}, {ps2}")

	homogeneity = evaluator.homogeneity_score(y_true, y_pred)
	completeness  = evaluator.CS(y_true, y_pred)
	print(f"Homogeneity: {homogeneity}, Completeness : {completeness}")


+ Internal clustering metrics

.. code-block:: python
	:emphasize-lines: 2,9,11,12

	import numpy as np
	from permetrics import ClusteringMetric
	from sklearn.datasets import make_blobs

	# generate sample data
	X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
	y_pred = np.random.randint(0, 4, size=300)

	evaluator = ClusteringMetric()

	evaluator.BHI(X=X, y_pred=y_pred)
	evaluator.BRI(X=X, y_pred=y_pred)


Object-Oriented Style
---------------------

+ External clustering metrics

.. code-block:: python
	:emphasize-lines: 2,7,10-12

	import numpy as np
	from permetrics import ClusteringMetric

	y_true = [0, 1, 0, 0, 1, 0]
	y_pred = [0, 1, 0, 0, 0, 1]

	evaluator = ClusteringMetric(y_true, y_pred)

	## Get the result of any function you want to
	x1 = evaluator.kulczynski_score()
	x2 = evaluator.mc_nemar_score()
	x3 = evaluator.rogers_tanimoto_score()
	print(f"Kulczynski: {x1}, Mc Nemar: {x2}, Rogers Tanimoto: {x3}")


+ Internal clustering metrics

.. code-block:: python
	:emphasize-lines: 2,9,11-13,16

	import numpy as np
	from permetrics import ClusteringMetric
	from sklearn.datasets import make_blobs

	# generate sample data
	X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
	y_pred = np.random.randint(0, 4, size=300)

	evaluator = ClusteringMetric(X=X, y_pred=y_pred)

	evaluator.BHI()
	evaluator.CHI()
	evaluator.DBI()

	## Or
	print(evaluator.get_metrics_by_list_names(["BHI", "CHI", "XBI", "BRI", "DBI", "DRI", "DI", "KDI", "LDRI", "LSRI", "SI"]))
