NRMSE - Normalized Root Mean Square Error
=========================================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2

The **Normalized Root Mean Square Error (NRMSE)** facilitates the comparison of models across different datasets with different scales by normalizing the RMSE.
Because there is no single consensus on the normalization factor in statistical literature, `permetrics` explicitly implements
**four standard normalization methods**. To prevent score manipulation, all methods strictly utilize the ground truth values (:math:`y_{\text{true}}`) as the denominator.


.. math::

    \text{NRMSE} = \frac{\text{RMSE}}{\text{Normalization Factor}}

-------------------------------------------------------------------------------

Supported Normalization Methods
-------------------------------

You can select the normalization method via the ``normalization`` parameter:

1. **Mean (CV-RMSE):** ``normalization="mean"`` (Default)
   Divides the RMSE by the mean of the observed values. Widely adopted as an industrial standard in energy forecasting (e.g., ASHRAE Guideline 14).
2. **Range:** ``normalization="range"``
   Divides by the difference between the maximum and minimum observed values (:math:`y_{\text{max}} - y_{\text{min}}`).
3. **Standard Deviation:** ``normalization="std"``
   Divides by the standard deviation of the observed values (:math:`\sigma`).
4. **Interquartile Range:** ``normalization="iqr"``
   Divides by the difference between the 75th and 25th percentiles (:math:`Q3 - Q1`). Highly recommended when the dataset contains extreme outliers that distort the range or mean.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``0.0`` (Smaller value is better).
* **Range:** ``[0, +inf)``
* **Mathematical Reference:** `Link 1`_, `Link 2`_, `Link 3`_

.. _Link 1: https://www.marinedatascience.co/blog/2019/01/07/normalizing-the-rmse/
.. _Link 2: https://search.r-project.org/CRAN/refmans/hydroGOF/html/nrmse.html
.. _Link 3: https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalized_root-mean-square_deviation

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 11-14, 23

    from numpy import array
    from permetrics.regression import RegressionMetric

    ## 1. For 1-D array (Single-output)
    y_true = array([3, -0.5, 2, 7])
    y_pred = array([2.5, 0.0, 2, 8])

    evaluator = RegressionMetric(y_true, y_pred)

    # Evaluate using different normalization factors
    print("NRMSE (Mean): ", evaluator.NRMSE(normalization="mean"))
    print("NRMSE (Range): ", evaluator.NRMSE(normalization="range"))
    print("NRMSE (Std): ", evaluator.NRMSE(normalization="std"))
    print("NRMSE (IQR): ", evaluator.NRMSE(normalization="iqr"))

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6], [1, 2], [2.1, 2.2], [3.4, 5.5]])
    y_pred = array([[0, 2], [-1, 2], [8, -5], [1.1, 1.9], [2.0, 2.3], [3.0, 4.2]])

    evaluator = RegressionMetric(y_true, y_pred)

    # Return an array of scores for each column using IQR normalization
    print("NRMSE (Multi-output): ", evaluator.NRMSE(normalization="iqr", multi_output="raw_values"))
