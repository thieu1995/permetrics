Normalized Mutual Information (NMI)
===================================

.. toctree::
   :maxdepth: 3
   :caption: Normalized Mutual Information (NMI)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


Normalized Mutual Information (NMI) is defined as the ratio of the mutual information between two clusterings and the geometric mean of the entropies
of the two clusterings. It is a measure of the similarity between two clusterings of the same set of data points.

Mathematically, given two clusterings A and B of a set of data points X, NMI is defined as math::

	$$NMI(A,B) = \frac{2I(A,B)}{H(A) + H(B)}$$

where I(A,B) is the mutual information between clusterings A and B, and H(A) and H(B) are the entropies of clusterings A and B, respectively.

Mutual information between two clusterings A and B is defined as::

	$$I(A,B) = \sum_{a\in A} \sum_{b\in B} p(a,b) \log \frac{p(a,b)}{p(a)p(b)}$$

where p(a,b) is the joint probability of data points belonging to clusters a in A and b in B, and p(a) and p(b) are the
marginal probabilities of data points belonging to clusters a in A and b in B, respectively.

The entropy of a clustering A is defined as::

	$$H(A) = - \sum_{a\in A} p(a) \log p(a)$$

where p(a) is the probability of data points belonging to cluster a in A. An example to illustrate the calculation of NMI:

Suppose we have a set of data points X and two clusterings A and B, where A has three clusters {A1, A2, A3} and B has two clusters {B1, B2}.
The number of data points in each cluster is shown in the table below:

+--------+-------------+---------+---------+--------+--------+
| Cluster|A1           |A2       |A3       |B1      |B2      |
+========+=============+=========+=========+========+========+
| # of data points|5      |7       |8       |9       |4       |
+--------+-------------+---------+---------+--------+--------+

The marginal probability p(a) can be calculated as the number of data points belonging to a divided by the total number of data points:

+--------------+----------+----------+----------+----------+
|Probabilities |p(A1)     |p(A2)     |p(A3)     |p(B1)     |p(B2)     |
+==============+==========+==========+==========+==========+==========+
|Value         |5/24      |7/24      |8/24      |9/24      |4/24      |
+--------------+----------+----------+----------+----------+

With these probabilities, we can calculate the mutual information I(A,B), the entropy H(A), and the entropy H(B) as follows::

	{p(a,b)}{p(a)p(b)} = (5/24)\log\frac{5/24}{(5/24)(9/24)} + (0)\log\frac{0}{(5/24)(4/24)} + (7/24)\log\frac{7/24}{(7/24)(9/24)} + (1/24)\log\frac{1/24}{(7/24)(4/24)} + (4/24)\log\frac{4/24}{(8/24)(9/24)} + (4/24)\log\frac{4/24}{(8/24)(4/24)}

	$$H(A) = - \sum_{a\in A} p(a) \log p(a) = -(5/24)\log(5/24) - (7/24)\log(7/24) - (8/24)\log(8/24)$$

	$$H(B) = - \sum_{b\in B} p(b) \log p(b) = -(9/24)\log(9/24) - (4/24)\log(4/24)$$

Finally, we can calculate NMI as::

	$$NMI(A,B) = \frac{2I(A,B)}{H(A) + H(B)}$$

Note that the NMI score is normalized so that it is invariant to the number of data points and the number of clusters in each clustering.
A higher NMI score indicates a higher similarity between the two clusterings, while a lower NMI score indicates a lower similarity.


+ Best possible score is 1.0, higher value is better. Range = [0, 1]
+ https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2
+ https://www.debadityachakravorty.com/ai-ml/cmatrix/
+ https://neptune.ai/blog/evaluation-metrics-binary-classification


Example:

.. code-block:: python
	:emphasize-lines: 11,13-16

	from numpy import array
	from permetrics.classification import ClassificationMetric

	## For integer labels or categorical labels
	y_true = [0, 1, 0, 0, 1, 0]
	y_pred = [0, 1, 0, 0, 0, 1]

	# y_true = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
	# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]

	cm = ClassificationMetric(y_true, y_pred, decimal = 5)

	print(cm.accuracy_score(average=None))
	print(cm.accuracy_score(average="micro"))
	print(cm.AS(average="macro"))
	print(cm.AS(average="weighted"))

