RSI - R-Squared Index
=====================

.. toctree::
   :maxdepth: 3

.. contents:: Table of Contents
   :local:
   :depth: 2


The **R-Squared Index (RSI)** (also known as the **Coefficient of Determination for Clustering**) is an internal evaluation metric. It measures the proportion of the total variance in the dataset that is explained by the clustering partition.

Intuitively, RSI acts similarly to the R-squared score in linear regression models. It answers the question: *"How much of the total dispersion of the data is captured by grouping the points into these clusters?"* A higher RSI value indicates a more optimal partition, implying that clusters are tightly cohesive and capture the vast majority of the dataset's variation.

.. math::

    \text{RSI} = \frac{\text{TSS} - \text{WGSS}}{\text{TSS}} = \frac{\text{BGSS}}{\text{TSS}}

Where:

* :math:`\text{TSS}` is the Total Sum of Squares (total dispersion of all data points around the global dataset centroid).
* :math:`\text{WGSS}` is the Within-Group Sum of Squares (total dispersion of data points around their respective cluster centroids).
* :math:`\text{BGSS}` is the Between-Group Sum of Squares (:math:`\text{TSS} - \text{WGSS}`).

-------------------------------------------------------------------------------

Properties
----------

* **Best possible score:** ``1.0`` (Higher value is better, indicating that 100% of the data variance is explained by the cluster centers).
* **Worst possible score:** ``0.0`` (Indicates the clustering captures zero variance, performing no better than a single global mean).
* **Range:** ``[0.0, 1.0]``

-------------------------------------------------------------------------------

Example Usage
-------------

.. code-block:: python
    :emphasize-lines: 12,13,22,23

    from permetrics.clustering import ClusteringMetric
    import numpy as np

    # ==============================================================================
    # SCENARIO 1: Basic Evaluation
    # ==============================================================================
    print("--- 1. BASIC R-SQUARED INDEX EXAMPLE ---")

    X_data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    y_pred_labels = np.array([0, 0, 0, 1, 1, 1])

    cm = ClusteringMetric(X=X_data, y_pred=y_pred_labels)
    rsi_score = cm.RSI()
    print(f"R-Squared Index: {rsi_score}")

    # ==============================================================================
    # SCENARIO 2: Single Cluster Evaluation (Zero variance explained)
    # ==============================================================================
    print("\n--- 2. SINGLE CLUSTER EXAMPLE ---")

    y_pred_single = np.array([0, 0, 0, 0, 0, 0])
    cm_single = ClusteringMetric(X=X_data, y_pred=y_pred_single)
    rsi_single = cm_single.RSI()
    print(f"RSI with 1 cluster: {rsi_single}")
