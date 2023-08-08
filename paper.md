---
title: 'PerMetrics: A Framework of Performance Metrics for Machine Learning Models'
tags:
  - performance metrics
  - classification 
  - regression
  - clustering
authors:
  - name: Nguyen Van Thieu
    orcid: 0000-0001-9994-8747
    corresponding: true
    affiliation: 1
affiliations:
  - name: Faculty of Computer Science, Phenikaa University, Yen Nghia, Ha Dong, Hanoi, 12116, Vietnam.
    index: 1
date: 08 Aug 2023
bibliography: paper.bib
---

# Summary

Performance metrics are pivotal in machine learning field, especially for tasks like regression, classification, and clustering [@saura_using_2021]. They offer quantitative measures to assess the accuracy and efficacy of models, aiding researchers and practitioners in evaluating, contrasting, and enhancing algorithms and models.
In regression tasks, where continuous predictions are made, metrics like MSE, RMSE, and Coefficient of Determination [@nguyen2018resource; @nguyen2019building] can reveal how well models capture data patterns. In classification tasks, metrics such as accuracy, precision, recall, F1-score, and AUC-ROC [@luque_impact_2019] assess a model's ability to classify instances correctly, detect false results, and gauge overall predictive performance. Clustering tasks aim to discover inherent patterns and structures within unlabeled data by grouping similar instances together. Metrics like Silhouette coefficient, Davies-Bouldin index, and Calinski-Harabasz index [@nainggolan_improved_2019] measure clustering quality, helping evaluate how well algorithms capture data distribution and assign instances to clusters.
In general, performance metrics serve multiple purposes. They enable researchers to compare different models and algorithms [@ahmed2021comprehensive], identify strengths and weaknesses [@Nguyen2019], and make informed decisions about model selection and parameter tuning [@nguyen2020new]. Moreover, it also plays a crucial role in the iterative process of model development and improvement. By quantifying the model's performance, metrics guide the optimization process [@nguyen2020eo], allowing researchers to fine-tune algorithms [@nguyen2021multi], explore feature engineering techniques, and address issues such as overfitting, underfitting, and bias [@thieu_groundwater_2023].
This paper introduces a Python framework named PerMetrics (PERformance METRICS), designed to offer comprehensive performance metrics for machine learning models. The library, referred to as permetrics, is open-source and written in Python. It provides a wide number of metrics to enable users to evaluate their models effectively. PerMetrics is hosted on GitHub and is under continuous development and maintenance by the dedicated team. The framework is accompanied by comprehensive documentation, examples, and test cases, facilitating easy comprehension and integration into users' workflows.


# Statement of need

Permetrics is a Python project developed in the field of performance assessment and machine learning. To the best of our knowledge, it is the first open-source framework that contributes a significant number of metrics, totaling 103 methods, for three fundamental problems: regression, classification, and clustering. This library relies exclusively on only two well-known third-party Python scientific computing packages: NumPy [@harris2020array] and SciPy [@virtanen2020scipy]. The modules of permetrics are extensively documented, and the automatically generated API provides a complete and up-to-date description of both the object-oriented and functional implementations underlying the framework.

To gain a better understanding of the necessity of permetrics library, this section will compare it to several notable libraries currently are available. Most notably, Scikit-Learn [@scikit_learn], which also encompasses an assortment of metrics for regression, classification, and clustering problems. Nevertheless, a few classification metrics present in Scikit-Learn lack support for multiple outputs, such as the Matthews correlation coefficient (MCC) and Hinge loss. Furthermore, critical metrics such as root mean square error, mean absolute percentage error, NSE, and KGE are absent. PerMetrics addresses these deficiencies. Additionally, Scikit-Learn is deficient in various vital clustering metrics, including but not limited to Ball Hall index, Banfeld Raftery index, sum of squared error, Duda Hart index, and Hartigan index [@desgraupes2013clustering].

Another popular package is Metrics [@benhamner]. It provides a variety of metrics for different programming languages such as Python, MATLAB, R, and Haskell. However, the development team has ceased activity since 2015. They offers a limited number of metrics because they focused on creating a single set of metrics for multiple languages. Additionally, the metrics are not packaged as a complete library but rather exist as repository code on GitHub.

TorchMetrics [@torchmetrics] is a widely recognized framework for performance metrics developed for PyTorch users. The library includes over 100 metrics, covering various domains such as regression, classification, audio, detection, and text. However, TorchMetrics does not provide metrics specifically for clustering tasks. Although it offers a substantial number of metrics, it falls short compared to permetrics. Moreover, it relies heavily on other major libraries such as NumPy, Torch, Typing-extensions, Packaging, and Lightning-utilities. Additionally, using this library may not be easy for beginners in Python programming, as it requires a deep understanding of the Torch library to utilize TorchMetrics effectively.

Other popular libraries such as TensorFlow [@abadi2016tensorflow], Keras [@chollet2015keras], CatBoost [@prokhorenkova2018catboost], and MxNet [@chen2015mxnet] also contain modules dedicated to metrics. However, the issue with these libraries is that their metric modules are specific to each respective one. It is challenging to combine metric modules from different libraries with each other. If it is possible to combine them, it often requires installing numerous related libraries. Furthermore, the metric modules within each library are tailored to users who are familiar with that specific one, requiring users to learn multiple libraries, syntax structures, and necessary commands associated with each framework to use them in combination. These are significant obstacles when using metrics from such libraries.

All the aforementioned challenges are addressed by our permetrics library. It not only offers a simple and concise syntax and usage but also does not require any knowledge of other major libraries such as TensorFlow, Keras, or PyTorch. Additionally, it can be seamlessly integrated with any computational or machine learning library. In the future, we plan to expand permetrics to include other domains such as text metrics, audio metrics, detection metrics, and image metrics.

Test paper

# References
