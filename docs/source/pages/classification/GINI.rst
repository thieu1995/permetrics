GINI Index
==========

.. toctree::
   :maxdepth: 3
   :caption: GINI Index (GINI)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


The Gini index is a measure of impurity or inequality often used in decision tree algorithms for evaluating the quality of a split.
It quantifies the extent to which a split divides the target variable (class labels) unevenly across the resulting branches.

The Gini index ranges from 0 to 1, where 0 indicates a perfect split, meaning all the samples in each branch belong to the same class, and 1 indicates an
impure split, where the samples are evenly distributed across all classes. To calculate the Gini index, you can use the following formula:

Gini index = 1 - (sum of squared probabilities of each class)

For a binary classification problem, with two classes (0 and 1), the Gini index can be calculated as::

	Gini index = 1 - (p0^2 + p1^2)
	where p0 is the probability of class 0 and p1 is the probability of class 1 in the split.

For a multiclass classification problem, the Gini index is calculated as::

	Gini index = 1 - (p0^2 + p1^2 + ... + pn^2)
	where p0, p1, ..., pn are the probabilities of each class in the split.

The Gini index is used to evaluate the quality of a split and guide the decision tree algorithm to select the split that results in the lowest Gini index.

It's important to note that the Gini index is not typically used as an evaluation metric for the overall performance of a classification model. Instead, it
is primarily used within the context of decision trees for determining the optimal splits during the tree-building process. The Gini index is also used as a
metric to evaluate the performance of a binary classification model. It is a measure of how well the model separates the positive and negative classes.

+ Smaller is better (Best = 0), Range = [0, +1]

Example:

.. code-block:: python
	:emphasize-lines: 11-14

	from numpy import array
	from permetrics.classification import ClassificationMetric

	## For integer labels or categorical labels
	y_true = [0, 1, 0, 0, 1, 0]
	y_pred = [0, 1, 0, 0, 0, 1]

	# y_true = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
	# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]

	cm = ClassificationMetric(y_true, y_pred)
	print(cm.gini_index(average=None))
	print(cm.GINI()
	print(cm.GINI()
