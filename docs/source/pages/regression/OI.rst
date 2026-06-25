OI - Overall Index
==================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **Overall Index (OI)** :cite:`almodfer2022modeling` is a robust composite metric that evaluates the predictive accuracy of a model by simultaneously synthesizing two distinct types of errors: a normalized absolute error and a relative variance indicator.

.. math::

   \text{OI}(y, \hat{y}) = \frac{1}{2} \left[ 1 - \frac{\text{RMSE}(y, \hat{y})}{y_{max} - y_{min}} + \text{EC}(y, \hat{y}) \right]

Where:

* :math:`\text{RMSE}` is the Root Mean Square Error.
* :math:`y_{max} - y_{min}` is the range of the actual ground truth values.
* :math:`\text{EC}` is the Efficiency Coefficient (mathematically identical to Nash-Sutcliffe Efficiency or :math:`R2`).

-------------------------------------------------------------------------------

Description
-----------

**Key Insight: The Composite Advantage**
OI is highly effective because it balances two perspectives. The term :math:`\frac{\text{RMSE}}{y_{max} - y_{min}}` represents the normalized magnitude of the error (Scatter Index), while :math:`\text{EC}` captures the model's ability to reproduce the variability of the data. By combining them, OI prevents a model from achieving a high score if it only performs well in one aspect but fails in the other.

**Advantages:**
	* **Comprehensive Evaluation:** It offers a single, normalized "scorecard" value that is extremely useful for ranking multiple algorithms without needing to cross-reference RMSE and R2 separately.
	* **Scale-Independence:** Both core terms inside the equation are dimensionless, meaning OI can be safely used to compare model performance across completely different datasets, scales, and measurement units.

**Disadvantages:**
	* **The Zero-Variance Trap (Critical Flaw):** If all values in the ground truth dataset are identical, :math:`y_{max} - y_{min} = 0`, causing a fatal division-by-zero error. Furthermore, the EC calculation will also crash under zero variance.
	* **Complex Interpretation:** Unlike MAE or MAPE, a score of ``0.65`` does not have a direct physical or percentage-based translation. It is strictly a comparative index.

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Indicates a perfect RMSE of 0 and a perfect EC of 1).
* **Range:** ``(-inf, 1.0]``

-------------------------------------------------------------------------------

Example Usage
-------------

*Note: Ensure your ground truth dataset has variance (max != min) to avoid division by zero.*

.. code-block:: python
    :emphasize-lines: 10, 18

    from numpy import array
    from permetrics.regression import RegressionMetric

    ## 1. For 1-D array (Single-output)
    y_true = array([3, -0.5, 2, 7])
    y_pred = array([2.5, 0.0, 2, 8])

    evaluator = RegressionMetric(y_true, y_pred)
    # Calculate Overall Index
    print("OI: ", evaluator.OI())

    ## 2. For > 1-D array (Multi-output)
    y_true = array([[0.5, 1], [-1, 1], [7, -6]])
    y_pred = array([[0, 2], [-1, 2], [8, -5]])

    evaluator = RegressionMetric(y_true, y_pred)
    # Return an array of scores for each column
    print("OI (Multi-output): ", evaluator.OI(multi_output="raw_values"))
