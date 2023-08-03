Cohen Kappa Score (CKS)
=======================

.. toctree::
   :maxdepth: 3
   :caption: Cohen Kappa Score (CKS)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


The Cohen's Kappa score is a statistic that measures the level of agreement between two annotators on a categorical classification problem.
It is a measure of inter-annotator reliability that is often used in medical diagnoses, quality control, and content analysis.

The Kappa score is calculated as the ratio of the observed agreement between two annotators to the agreement that would be expected by chance.
The observed agreement is the number of instances that are classified the same way by both annotators, and the expected agreement is
the number of instances that are classified the same way by chance, given the individual annotator's classifications.

The formula for the Cohen's Kappa score is as follows .. math::

	k = (observed agreement - expected agreement) / (1 - expected agreement)

where observed agreement is the proportion of items that are classified the same way by both annotators, and expected
agreement is theproportion of items that are classified the same way by chance.

.. math::

	$\kappa = \frac{p_o - p_e}{1 - p_e}$

	where

	$p_o = \frac{tp + tn}{tp + tn + fp + fn}$

	$p_e = \frac{(tp + fp) \cdot (tp + fn) + (tn + fn) \cdot (tn + fp)}{(tp + tn + fp + fn)^2}$

	$tp$ represents the number of true positive annotations (agreements between the two annotators)

	$tn$ represents the number of true negative annotations (agreements between the two annotators)

	$fp$ represents the number of false positive annotations (disagreements between the two annotators)

	$fn$ represents the number of false negative annotations (disagreements between the two annotators)

	$p_o$ represents the observed agreement, and $p_e$ represents the expected agreement based on chance.


+ Best possible score is 1.0, higher value is better. Range = [-1, 1]
+ The value of k ranges from -1 to 1, with values closer to 1 indicating high levels of agreement, and values closer to -1 indicating low levels of agreement.
A value of 0 indicates that the agreement between the annotators is no better than chance. A value of 1 indicates perfect agreement.

The Cohen's Kappa score can be used for both binary and multi-class classification problems. For multi-class classification problems,
the observed agreement and expected agreement are calculated based on a confusion matrix, which is a table that shows the
number of instances that are classified into each possible pair of true and predicted classes.
The confusion matrix is used to calculate the observed agreement and expected agreement between the annotators, and
the resulting values are used in the formula for the Cohen's Kappa score.


It's important to note that the Cohen's Kappa score can be negative if the agreement between y_true and y_pred is lower than what would be expected by chance.
A value of 1.0 indicates perfect agreement, and a value of 0.0 indicates no agreement beyond chance.

Also, this implementation of the Cohen's Kappa score is flexible and can handle binary as well as multi-class classification problems.
The calculation of the confusion matrix and the subsequent calculation of the expected and observed agreements is based on the assumption
that the ground truth labels and predicted labels are integer values that represent the different classes.
If the labels are represented as strings or some other data type, additional pre-processing would be required to convert them
to integer values that can be used in the confusion matrix.



Example:

.. code-block:: python
	:emphasize-lines: 11,13-16

	from numpy import array
	from permetrics.classification import ClassificationMetric

	## For integer labels or categorical labels
	y_true = [0, 1, 2, 0, 2, 1, 1, 2, 2, 0]
	y_pred = [0, 0, 2, 0, 2, 2, 1, 1, 2, 0]

	# y_true = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
	# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]

	cm = ClassificationMetric(y_true, y_pred, decimal = 5)

	cm = ClassificationMetric(y_true, y_pred, decimal=5)

	print(cm.cohen_kappa_score(average=None))
	print(cm.CKS(average="micro"))
	print(cm.CKS(average="macro"))
	print(cm.CKS(average="weighted"))
