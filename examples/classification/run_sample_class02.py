from permetrics.classification import ClassificationMetric

# ==============================================================================
# SCENARIO 1: Binary Classification
# The default 'binary' mode requires a specific positive class (pos_label)
# ==============================================================================
print("--- 1. BINARY CLASSIFICATION EXAMPLES ---")

y_true_bin = [0, 1, 0, 0, 1, 0]
y_pred_bin = [0, 1, 0, 0, 0, 1]
cm_bin = ClassificationMetric(y_true_bin, y_pred_bin)

# 1. Default configuration: average="binary", pos_label=1
ps_bin_default = cm_bin.PS()
print(f"Default (average='binary', pos_label=1): {ps_bin_default}")

# 2. Change pos_label to 0 (treats 0 as the positive class)
ps_bin_pos0 = cm_bin.PS(average="binary", pos_label=0)
print(f"Binary with pos_label=0                : {ps_bin_pos0}")

# 3. When average=None, it returns independent scores for each class found
ps_bin_none = cm_bin.PS(average=None)
print(f"Binary with average=None               : {ps_bin_none}")

# ==============================================================================
# SCENARIO 2: Multiclass Classification with Integer Labels
# ==============================================================================
print("\n--- 2. MULTICLASS (INTEGER LABELS) EXAMPLES ---")

y_true_multi_int = [0, 1, 2, 0, 1, 2, 0, 2]
y_pred_multi_int = [0, 2, 1, 0, 1, 1, 0, 2]
cm_multi_int = ClassificationMetric(y_true_multi_int, y_pred_multi_int)

print(f"average=None       : {cm_multi_int.PS(average=None)}")
print(f"average='macro'    : {cm_multi_int.PS(average='macro')}")
print(f"average='micro'    : {cm_multi_int.PS(average='micro')}")
print(f"average='weighted' : {cm_multi_int.PS(average='weighted')}")

# Using the `labels` parameter to filter specific classes
print(f"Filter classes [1, 2] (average=None)      : {cm_multi_int.PS(labels=[1, 2], average=None)}")
print(f"Filter classes [1, 2] (average='macro')   : {cm_multi_int.PS(labels=[1, 2], average='macro')}")
print(f"Filter classes [1, 2] (average='micro')   : {cm_multi_int.PS(labels=[1, 2], average='micro')}")
print(f"Filter classes [1, 2] (average='weighted'): {cm_multi_int.PS(labels=[1, 2], average='weighted')}")

# ==============================================================================
# SCENARIO 3: Multiclass Classification with Categorical/String Labels
# ==============================================================================
print("\n--- 3. MULTICLASS (CATEGORICAL/STRING LABELS) EXAMPLES ---")

y_true_str = ["cat", "ant", "cat", "cat", "ant", "bird", "bird", "bird"]
y_pred_str = ["ant", "ant", "cat", "cat", "ant", "cat", "bird", "ant"]
cm_str = ClassificationMetric(y_true_str, y_pred_str)

print(f"average=None (Class dict) : {cm_str.PS(average=None)}")
print(f"average='macro'           : {cm_str.PS(average='macro')}")
print(f"average='micro'           : {cm_str.PS(average='micro')}")
print(f"average='weighted'        : {cm_str.PS(average='weighted')}")

# Filter string labels: Focus calculation entirely on 'cat' and 'bird'
print(f"Filter 'cat' & 'bird' (average=None)      : {cm_str.PS(labels=['cat', 'bird'], average=None)}")
print(f"Filter 'cat' & 'bird' (average='macro')   : {cm_str.PS(labels=['cat', 'bird'], average='macro')}")
print(f"Filter 'cat' & 'bird' (average='micro')   : {cm_str.PS(labels=['cat', 'bird'], average='micro')}")
print(f"Filter 'cat' & 'bird' (average='weighted'): {cm_str.PS(labels=['cat', 'bird'], average='weighted')}")
