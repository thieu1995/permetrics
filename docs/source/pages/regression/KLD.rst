KLD - Kullback-Leibler Divergence
=================================

.. toctree::
   :maxdepth: 3
   :caption: KLD - Kullback-Leibler Divergence

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


The Kullback-Leibler Divergence (KLD), :cite:`hershey2007approximating` also known as relative entropy, is a statistical measure of how different two probability
distributions are from each other. It was introduced by Solomon Kullback and Richard Leibler in 1951. The KLD is calculated as the sum of the logarithmic
differences between the probabilities of each possible outcome, weighted by the probability of the outcome in the reference distribution. The KLD is always
non-negative, and it is equal to zero if and only if the two distributions are identical. The equation for KLD between two probability distributions P and Q
is given by:

.. math::

    D_{KL}(P||Q) = \sum_{i} P(i) \log\frac{P(i)}{Q(i)}

where P(i) and Q(i) are the probabilities of the i-th possible outcome in the two distributions, respectively.

The KLD measures the information lost when approximating one probability distribution by another. It is widely used in information theory, machine learning,
and data science applications, such as clustering, classification, and data compression. The KLD has also found applications in other fields, such as
physics, economics, and biology, to measure the distance between two probability distributions.

+ Best possible score is 0.0 . Range = (-inf, +inf)
+ `Link to equation <https://machinelearningmastery.com/divergence-between-probability-distributions/>`_


Example to use KLD metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.kullback_leibler_divergence())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6], [1, 2], [2.1, 2.2], [3.4, 5.5]])
	y_pred = array([[0, 2], [-1, 2], [8, -5], [1.1, 1.9], [2.0, 2.3], [3.0, 4.2]])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.KLD(multi_output="raw_values"))


