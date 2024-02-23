G-Mean Score (GMS)
==================

.. toctree::
   :maxdepth: 3
   :caption: G-Mean Score (GMS)

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


G-mean is a performance metric in the field of machine learning and specifically in binary classification problems.
It is a balanced version of the geometric mean, which is calculated as the square root of the product of true positive rate (TPR) and true negative rate
(TNR) also known as sensitivity and specificity, respectively.

The G-mean is a commonly used metric to evaluate the performance of a classifier in imbalanced datasets where one class has a much higher
number of samples than the other. It provides a balanced view of the model's performance as it penalizes low values of TPR and TNR in a single score.
The G-mean score provides a balanced evaluation of a classifier's performance by considering both the positive and negative classes.

The formula for the G-mean score is given by

.. math::

	G-mean = sqrt(TPR * TNR)

	Gmean = \sqrt{TPR * TNR}

where TPR (True Positive Rate) is defined as

.. math::

	$TPR = \frac{TP}{TP + FN}$

and TNR (True Negative Rate) is defined as

.. math::

	$TNR = \frac{TN}{TN + FP}$

with TP (True Positives) as the number of instances that are correctly classified as positive, TN (True Negatives) as the number
of instances that are correctly classified as negative, FP (False Positives) as the number of instances that are wrongly classified
as positive, and FN (False Negatives) as the number of instances that are wrongly classified as negative.


+ Best possible score is 1.0, higher value is better. Range = [0, 1]
For a binary classification problem with two classes, the G-mean score provides a single value that represents the overall accuracy of the classifier.
A G-mean score of 1.0 indicates perfect accuracy, while a score of less than 1.0 indicates that one of the classes is being misclassified more frequently than the other.

In a multi-class classification problem, the G-mean score can be calculated for each class and then averaged over all classes to provide a
single value that represents the overall accuracy of the classifier. The average can be weighted or unweighted, depending on the desired interpretation of the results.

For example, consider a multi-class classification problem with three classes: class A, class B, and class C. The G-mean score for each class can be
calculated using the formula above, and then averaged over all classes to provide an overall G-mean score for the classifier.

The G-mean score provides a way to balance the accuracy of a classifier between positive and negative classes, and is particularly useful in
cases where the class distribution is imbalanced, or when one class is more important than the other.


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

	cm = ClassificationMetric(y_true, y_pred)

	print(cm.g_mean_score(average=None))
	print(cm.GMS(average="micro"))
	print(cm.GMS(average="macro"))
	print(cm.GMS(average="weighted"))

