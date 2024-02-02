JSD - Jensen-Shannon Divergence
===============================

.. toctree::
   :maxdepth: 3
   :caption: JSD - Jensen-Shannon Divergence

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3


The Jensen-Shannon Divergence (JSD) :cite:`fuglede2004jensen` is a statistical measure of the similarity between two probability distributions. It is named
after Danish mathematician Johan Jensen and American mathematician Richard Shannon, who introduced the concept in 1991.

The JSD is a symmetric and smoothed version of the Kullback-Leibler Divergence (KLD), which is a measure of how much one probability distribution differs
from another. Unlike the KLD, the JSD is always a finite value and satisfies the triangle inequality, making it a proper metric.

The JSD is calculated as follows:

+ Compute the average probability distribution by taking the arithmetic mean of the two distributions:

.. math::

    M = 0.5 * (P + Q)

where P and Q are the two probability distributions being compared.

+ Calculate the KLD between each distribution and the average distribution:

.. math::

    D(P || M) = \sum_{i} P(i) \log\frac{P(i)}{M(i)}

    D(Q || M) = \sum_{i} Q(i) \log\frac{Q(i)}{M(i)}

+ Compute the JSD as the arithmetic mean of the two KLD values:

.. math::

    JSD(P || Q) = \frac{1}{2} \left(D(P || M) + D(Q || M)\right)

The JSD measures the distance between two probability distributions, with values ranging from 0 (when the distributions are identical) to 1 (when the
distributions are completely different). It is commonly used in machine learning and information retrieval applications, such as text classification and clustering.

+ Best possible score is 0.0 (identical), smaller value is better . Range = [0, +inf)
+ `Link to equation <https://machinelearningmastery.com/divergence-between-probability-distributions/>`_


Example to use JSD metric:

.. code-block:: python
	:emphasize-lines: 8-9,15-16

	from numpy import array
	from permetrics.regression import RegressionMetric

	## For 1-D array
	y_true = array([3, -0.5, 2, 7])
	y_pred = array([2.5, 0.0, 2, 8])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.jensen_shannon_divergence())

	## For > 1-D array
	y_true = array([[0.5, 1], [-1, 1], [7, -6], [1, 2], [2.1, 2.2], [3.4, 5.5]])
	y_pred = array([[0, 2], [-1, 2], [8, -5], [1.1, 1.9], [2.0, 2.3], [3.0, 4.2]])

	evaluator = RegressionMetric(y_true, y_pred)
	print(evaluator.JSD(multi_output="raw_values"))


