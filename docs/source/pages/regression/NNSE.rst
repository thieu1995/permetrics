NNSE - Normalized Nash-Sutcliffe Efficiency
===========================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Normalized Nash-Sutcliffe Efficiency (NNSE)** :cite:`ahmed2021comprehensive` is a bounded transformation of the standard Nash-Sutcliffe Efficiency (NSE) metric, widely used in hydrology and water resources engineering.

While standard NSE ranges from negative infinity to 1, NNSE maps this infinite domain into a strictly bounded range between 0 and 1. This transformation allows for safe multi-model or multi-site averaging without the risk of a single catastrophically poor prediction (with an NSE approaching :math:`-\infty`) heavily skewing the aggregate results.

.. math::

    \text{NNSE}(y, \hat{y}) = \frac{1}{2 - \text{NSE}(y, \hat{y})}

Note:
	* When NSE = 1, NNSE = 1.
	* When NSE = 0, NNSE = 0.5.
	* As NSE approaches :math:`-\infty`, NNSE approaches 0.

-------------------------------------------------------------------------------

Description
-----------

**Advantages:**
	* **Stable Aggregation (Crucial Feature):** Because the range is strictly bounded, you can safely calculate the mean or median NNSE across dozens of different catchment areas or evaluation periods.
	* **Bounded Optimization:** Highly useful as an objective/loss function in machine learning algorithms that require bounded metrics (where infinite values would cause gradient explosions).

**Disadvantages:**
	* **Resolution Compression:** The normalization heavily compresses negative NSE values. For instance, a slightly bad model (NSE = -2) yields an NNSE of 0.25, while a completely catastrophic model (NSE = -98) yields an NNSE of 0.01. It becomes difficult to differentiate the severity of "bad" models just by looking at their NNSE scores.
	* **Loss of Intuition:** The intuitive baseline of NSE = 0.0 (where the model is as good as the mean) is shifted to NNSE = 0.5, which can occasionally confuse stakeholders used to standard R2 or NSE scales.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates perfect agreement between observed and simulated data).
* **Baseline score:** ``0.5`` (Model is only as accurate as predicting the observed mean).
* **Range:** ``(0.0, 1.0]`` (Mathematically, it approaches 0 but never strictly reaches it unless the error is genuinely infinite).

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 10, 18

    from numpy import array
    from permetrics.regression import RegressionMetric

    ## 1. For 1-D array (Single-output)
    y_true = array([3, -0.5, 2, 7])
    y_pred = array([2.5, 0.0, 2, 8])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate Normalized Nash-Sutcliffe Efficiency
    print("NNSE: ", evaluator.NNSE())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6], [1, 2], [2.1, 2.2], [3.4, 5.5]])
    y_pred = array([[0, 2], [-1, 2], [8, -5], [1.1, 1.9], [2.0, 2.3], [3.0, 4.2]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("NNSE (Multi-output): ", evaluator.NNSE(multi_output="raw_values"))
