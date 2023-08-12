
# Version 1.4.3

+ Reformat ClassificationMetric, RegressionMetric, and ClusteringMetric
+ Fix bugs in metrics:
  + ROC-AUC score
  + F2 score, FBeta score
  + Hamming Score, Accuracy Score
+ Add metrics to ClassificationMetric:
  + Brier Score Loss (BSL)
  + Kullback-Leibler Divergence Loss (KLDL)
  + Cross-Entropy Loss (CEL)
  + Hinge Loss (HL)
+ Update docs, examples

---------------------------------------------------------------------

# Version 1.4.2

+ Fix bug in get_support() function
+ Fix bug rounding number in ClassificationMetric 
+ Update logo and docs

---------------------------------------------------------------------

# Version 1.4.1

+ Remove all lowercase shortname of all metrics
+ Fix bugs in GINI function belongs to ClassificationMetric
+ Fix bugs in some functions belong to ClusteringMetric
+ Update characteristics of most of the functions (valid ranges and best value)
+ Add metrics to ClusteringMetrics
  + Entropy Score (ES)
  + Purity Score (PuS)
  + Tau Score (TS)
  + Duda-Hart Index (DHI)
  + Sum of Squared Error Index (SSEI)
  + Beale Index (BI)
  + R-Squared Index (RSI)
  + Density-Based Clustering Validation Index (DBCVI)
  + Hartigan Index (HI)
+ Add get_support() method in RegressionMetric, ClassificationMetric, and ClusteringMetric
+ Update and add more examples to all classes.
+ Update documents for all metrics.

---------------------------------------------------------------------

# Version 1.4.0

+ Add ClusteringMetric:
  + Add internal metrics (Need X features and y_pred)
    + ball_hall_index (BHI)
    + calinski_harabasz_index (CHI)
    + xie_beni_index (XBI)
    + banfeld_raftery_index (BRI)
    + davies_bouldin_index (DBI)
    + det_ratio_index (DRI)
    + dunn_index (DI)
    + ksq_detw_index (KDI)
    + log_det_ratio_index (LDRI)
    + log_ss_ratio_index (LSRI)
    + silhouette_index (SI)
  + Add external metrics (Need y_true and y_pred)
    + mutual_info_score (MIS)
    + normalized_mutual_info_score (NMIS)
    + rand_score (RaS)
    + fowlkes_mallows_score (FMS)
    + homogeneity_score (HS)
    + completeness_score (CS)
    + v_measure_score (VMS)
    + precision_score (PrS)
    + recall_score (ReS)
    + f_measure_score (FmS)
    + czekanowski_dice_score (CDS)
    + hubert_gamma_score (HGS)
    + jaccard_score (JS)
    + kulczynski_score (KS)
    + mc_nemar_score (MNS)
    + phi_score (PhS)
    + rogers_tanimoto_score (RTS)
    + russel_rao_score (RRS)
    + sokal_sneath1_score (SS1S)
    + sokal_sneath2_score (SS2S)
+ Add examples for ClusteringMetric
+ Add LabelEncoder class 

---------------------------------------------------------------------

# Version 1.3.3

### Update

Update ClassificationMetric:
+ Rename confusion_matrix() in util file
+ Support binary and multi-class classification with one-hot-encoder format
+ Add Cohen's Kappa score 
+ Add Jaccard Similarity Index (Jaccard similarity coefficient)
+ Add G-mean score 
+ Add GINI index 
+ Add ROC-AUC metric 


---------------------------------------------------------------------


# Version 1.3.2

### Update

Add regression metrics:
+ covariance (COV): https://corporatefinanceinstitute.com/resources/data-science/covariance/
+ correlation (COR): https://corporatefinanceinstitute.com/resources/data-science/covariance/
+ efficiency coefficient (EC): https://doi.org/10.1016/j.solener.2019.01.037
+ overall index (OI): https://doi.org/10.1016/j.solener.2019.01.037
+ coefficient of residual mass (CRM): https://doi.org/10.1016/j.solener.2019.01.037


---------------------------------------------------------------------


# Version 1.3.1

### Update

Add classification metrics: 
+ Fix bug in MCC metric
+ Fix bug exit() program
+ Update documents
+ Update examples

---------------------------------------------------------------------


# Version 1.3.0

### Update

Add classification metrics: 
+ Add Precision Score (PS)
+ Add Negative Predictive Value (NPV)
+ Add Recall Score (RS)
+ Add Accuracy Score (AS)
+ Add F1 Score (F1S)
+ Add F2 Score (F2S)
+ Add F Beta Score (FBS)
+ Add Specificity Score (SS)
+ Add Matthews Correlation Coefficient (MCC)
+ Add Hamming Loss (HL)
+ Add Lift Score (LS)

+ Update reference documents
+ Update examples

---------------------------------------------------------------------

# Version 1.2.2

### Update

+ Add Absolute R metric (AR) 
+ Add A30 metric (A30)
+ Add Mean Bias Error (MBE)
+ Add Mean Percentage Error (MPE)
+ Add Adjusted R2 (AR2)
+ Update reference documents
+ Update examples

---------------------------------------------------------------------


# Version 1.2.1

### Update

+ Fix bugs copy original data from Evaluator class 
+ Fix bugs of some methods in Regression class 
+ Add utils module
+ Update reference documents
+ Update examples

---------------------------------------------------------------------

# Version 1.2.0

### Change models

+ Add Evaluator class, the base class for all metrics
+ Rename class Metrics in regression to RegressionMetric
+ Rename class Metrics in classification to ClassificationMetric
+ Update input parameters in both RegressionMetric and ClassificationMetric
+ Update input parameters in all metric functions
+ Add docstring for all methods
+ Merge all singleloss metrics to RegressionMetric
+ Update examples
+ Add new documents and website

---------------------------------------------------------------------

# Version 1.1.1 / 1.1.2 / 1.1.3 

### Fix bugs

+ Fix bugs of R2S, MAPE and some other functions

---------------------------------------------------------------------

# Version 1.1.0

### Change models

+ Change class Metrics in regression.py to take both OOP style and Functional style. 
+ Add some new metrics including
23. PCD: Prediction of Change in Direction
24. E: Entropy
25. CE: Cross Entropy
26. KLD: Kullback Leibler Divergence
27. JSD: Jensen Shannon Divergence
28. VAF: Variance Accounted For
29. RAE: Relative Absolute Error
30. A10: A10 Index
31. A20: A20 Index
32. NRMSE: Normalized root Mean Square Error
33. RSE: Residual Standard Error

### Change others

+ Examples:
  + Update examples of old metrics
  + Add examples for new metrics

---------------------------------------------------------------------

# Version 1.0.4

### Change models
+ Add some new methods to regression.py
21. Gini coefficient
    + Based on other code (Mathlab code)
    + Based on wiki version
22. Mean Log Likelihood
    
+ Add new class SingleLoss for numbers-verse-numbers, the output is numbers (not single number as regression.py)
1. Relative error
2. Absolute error
3. Squared error
4. Squared log error
5. Log likelihood
    
+ Add new class for classification metrics
1. Mean Log Likelihood
    
    
### Change others
+ Examples: 
    + Add examples
    + Add documents

---------------------------------------------------------------------

# Version 1.0.2

### Change models
+ Add some new methods:
18. Kling-Gupta Efficiency (KGE)
19. Deviation of Runoff Volume (DRV)
20. (Pearson’s Correlation Index)**2 or R2s (or R2)
    
### Change others
+ Examples: 
    + Add examples
    + Add documents

---------------------------------------------------------------------

# Version 1.0.1 

### Change models
+ Add method for multiple metrics called at the same time
    + Call by list of function names (default function parameters)
    + Call by list of function names and parameters (can change function parameters)
### Change others
+ Examples: 
    + Add all examples for all metrics
    + Add example for multiple metrics called at the same time
+ Documents: Add all documents for all metrics
    
---------------------------------------------------------------------
# Version 1.0.0 (First version)

## Models

### Regression models, we have 17 metrics 
1. Explained Variance Score
2. Max Error
3. Mean Absolute Error
4. Mean Squared Error
5. Root Mean Squared Error
6. Mean Squared Log Error
7. Median Absolute Error
8. Mean Relative Error
9. Mean Absolute Percentage Error
10. Symmetric Mean Absolute Percentage Error
11. Mean Arctangent Absolute Percentage Error
12. Mean Absolute Scaled Error
13. Nash-Sutcliffe Efficiency Coefficient
14. Willmott Index
15. Pearson’s Correlation Index
16. Confidence Index 
17. Coefficient of Determination

