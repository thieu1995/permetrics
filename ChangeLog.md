
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

