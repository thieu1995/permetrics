Tau Score (TS)
==============

.. toctree::
   :maxdepth: 3
   :caption: Tau Score (TS)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

The Tau index, also known as the Tau coefficient, is a measure of agreement or similarity between two clustering solutions.
It is commonly used to compare the similarity of two different clusterings or to evaluate the stability of a clustering algorithm.

The Tau index is based on the concept of concordance, which measures the extent to which pairs of objects are assigned to
the same clusters in two different clustering solutions. The index ranges from -1 to 1, where 1 indicates perfect agreement,
0 indicates random agreement, and -1 indicates perfect disagreement or inversion of the clustering solutions.

The calculation of the Tau index involves constructing a contingency table that counts the number of pairs of objects
that are concordant (i.e., assigned to the same cluster in both solutions) and discordant
(i.e., assigned to different clusters in the two solutions).

The formula for calculating the Tau index is as follows::

   Tau = (concordant_pairs - discordant_pairs) / (concordant_pairs + discordant_pairs)

A higher value of the Tau index indicates greater similarity or agreement between the two clusterings,
while a lower value indicates less agreement. It's important to note that the interpretation of the Tau index
depends on the specific clustering algorithm and the data being clustered.

The Tau index can be useful in various applications, such as evaluating the stability of clustering algorithms,
comparing different clustering solutions, or assessing the robustness of a clustering method to perturbations
in the data. However, like any clustering evaluation measure, it has its limitations and should be used
in conjunction with other evaluation techniques to gain a comprehensive understanding of the clustering performance.

Example:

.. code-block:: python

	import numpy as np
	from permetrics import ClusteringMetric

	## For integer labels or categorical labels
	y_true = np.array([0, 0, 1, 1, 1, 2, 2, 1])
	y_pred = np.array([0, 0, 1, 1, 2, 2, 2, 2])

	cm = ClusteringMetric(y_true=y_true, y_pred=y_pred)

	print(cm.tau_score())
	print(cm.TS())
