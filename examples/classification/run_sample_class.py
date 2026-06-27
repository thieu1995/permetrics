#!/usr/bin/env python
# Created by "Thieu" at 00:00, 26/06/2026 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score, brier_score_loss
from permetrics import ClassificationMetric


def is_close_enough(x1, x2, eps=1e-5):
    if abs(x1 - x2) <= eps:
        return True
    return False


# ==============================================================================
# SCENARIO 1: Standard Discrete Targets (Hard Labels)
# y_pred expects a 2D matrix of forecasted class probabilities
# ==============================================================================
print("--- 1. STANDARD HARD TARGETS ---")

y_true_hard = [0, 1, 2]
y_pred_prob = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]

cm_hard = ClassificationMetric(y_true_hard, y_pred_prob)
print(f"Standard CEL : {cm_hard.CEL()}")

# ==============================================================================
# SCENARIO 2: Label Smoothing Regularization (Soft Targets)
# y_true is passed directly as a smoothed probability matrix
# ==============================================================================
print("\n--- 2. LABEL SMOOTHED SOFT TARGETS ---")

y_true_soft = [[0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.10, 0.10, 0.80]]

cm_soft = ClassificationMetric(y_true_soft, y_pred_prob)
print(f"Smoothed CEL : {cm_soft.CEL()}")



# y_true = [0, 1, 2, 3, 0, 1]
# y_pred = [0, 1, 2, 3, 1, 4]
#
# y_true = ["cat", "dog", "dog", "cat"]
# y_pred = ["cat", "dog", "dog", "dog"]
#
# # CHẾ ĐỘ MẶC ĐỊNH: labels=None
# print("Mặc định (labels=None):", precision_score(y_true, y_pred, average=None))
# # Kết quả chỉ trả về 4 số: [0.5, 0.5, 1.0, 1.0] -> Mất dấu class 4!
#
# # CHẾ ĐỘ CHỦ ĐỘNG: Định nghĩa rõ labels
# print("Chủ động (labels=[0,1,2,3,4]):", precision_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None))
# #
# cm = ClassificationMetric(y_true, y_pred)
# print(cm.PS(average=None))
# print(cm.PS(average="binary", pos_label="dog"))
#
# # # Tập dữ liệu nhị phân (0 và 1)
# # y_true = [0, 0, 1, 1, 1]
# # y_pred = [0, 1, 0, 1, 1]




# # 1. Chế độ mặc định (Chỉ tính cho lớp 1)
# print("Mặc định (average='binary'):", precision_score(y_true, y_pred))
# # Kết quả: 0.6666666666666666
#
# # 2. Muốn lấy điểm của cả 2 lớp riêng biệt
# print("Xem cả 2 lớp (average=None): ", precision_score(y_true, y_pred, average=None))
# # Kết quả: [0.5        0.66666667]
#
# # 3. Muốn lấy giá trị trung bình của 2 lớp
# print("Trung bình 2 lớp (average='macro'):", precision_score(y_true, y_pred, average='macro'))
# # Kết quả: 0.5833333333333333  (Chính là (0.5 + 0.6666) / 2)




binary_label_cases = {
    # # --- NORMAL CASES ---
    "normal_perfect": {
        "comment": "Perfect predictions. All metrics should be 1.0.",
        "y_true": np.array([0, 1, 0, 1, 1, 0]),
        "y_pred": np.array([0, 1, 0, 1, 1, 0])
    },
    "normal_mixed": {
        "comment": "Standard case with mixed TP, TN, FP, FN.",
        "y_true": np.array([0, 1, 0, 1, 1, 0, 1, 0]),
        "y_pred": np.array([0, 0, 1, 1, 1, 0, 0, 1])
    },
    "imbalanced_skewed": {
        "comment": "Highly imbalanced dataset. Tests if metrics handle rare positive class.",
        "y_true": np.array([0, 0, 0, 0, 0, 0, 0, 1]),
        "y_pred": np.array([0, 0, 0, 0, 0, 0, 0, 0])
    },

    "string_labels": {
        "comment": "Check if your library handles text/string labels like sklearn.",
        "y_true": np.array(["cat", "dog", "dog", "cat"]),
        "y_pred": np.array(["cat", "cat", "dog", "cat"])
    },
    # "mix_string_labels": {
    #     "comment": "Check if your library handles text/string labels like sklearn.",
    #     "y_true": np.array(["cat", "dog", "dog", "cat"]),
    #     "y_pred": np.array([0, 0, 1, 1])
    # },

    # --- EDGE CASES ---
    "all_zeros": {
        "comment": "No positive samples in true labels. Recall denominator is 0.",
        "y_true": np.array([0, 0, 0, 0]),
        "y_pred": np.array([0, 0, 0, 0])
    },
    "all_ones": {
        "comment": "No negative samples in true labels.",
        "y_true": np.array([1, 1, 1, 1]),
        "y_pred": np.array([1, 1, 1, 1])
    },
    "zero_predicted_positive": {
        "comment": "Model predicts NO positives. Precision denominator is 0.",
        "y_true": np.array([0, 1, 0, 1]),
        "y_pred": np.array([0, 0, 0, 0])
    },
    "zero_predicted_negative": {
        "comment": "Model predicts ONLY positives. Specificity denominator is 0.",
        "y_true": np.array([0, 1, 0, 1]),
        "y_pred": np.array([1, 1, 1, 1])
    },
    "completely_wrong": {
        "comment": "Inverted predictions. Accuracy and F1 should be 0.0.",
        "y_true": np.array([0, 1, 0, 1]),
        "y_pred": np.array([1, 0, 1, 0])
    },
    "single_element": {
        "comment": "Minimum possible array size.",
        "y_true": 1,
        "y_pred": [0.1, 0.9]
    }
}

binary_prob_cases = {
    # --- NORMAL CASES ---
    # "prob_normal": {
    #     "comment": "Standard probabilities. Good for Log Loss and ROC-AUC.",
    #     "y_true": np.array([0, 0, 1, 1]),
    #     "y_score": np.array([0.1, 0.4, 0.35, 0.8])
    # },

    "prob_normal_2D": {
        "comment": "Standard probabilities. Good for Log Loss and ROC-AUC.",
        "y_true": np.array([0, 0, 1, 1]),
        "y_score": np.array([[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]])
    },

    # --- EDGE CASES ---
    "prob_perfect": {
        "comment": "Perfect confidence. Log loss should be 0.0.",
        "y_true": np.array([0, 1, 0, 1]),
        "y_score": np.array([0.0, 1.0, 0.0, 1.0])
    },
    "prob_extreme_wrong": {
        "comment": "100% confident but completely wrong. Tests log(0) handling.",
        "y_true": np.array([0, 1]),
        "y_score": np.array([1.0, 0.0])  # Sklearn clips these to avoid infinity
    },
    "prob_all_constant": {
        "comment": "Model predicts same probability for all. ROC-AUC should be 0.5.",
        "y_true": np.array([0, 0, 1, 1]),
        "y_score": np.array([0.5, 0.5, 0.5, 0.5])
    },
    "prob_single_class_true": {
        "comment": "ROC-AUC should raise an error in sklearn because only 1 class exists in y_true.",
        "y_true": np.array([1, 1, 1]),
        "y_score": np.array([0.9, 0.8, 0.7])
    }
}

multiclass_label_cases = {
    # --- NORMAL CASES ---
    "multiclass_normal": {
        "comment": "Standard 3-class problem with mixed performance.",
        "y_true": np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
        "y_pred": np.array([0, 2, 2, 0, 1, 1, 0, 1, 2])
    },

    # --- EDGE CASES ---
    "missing_class_in_pred": {
        "comment": "Class 2 exists in y_true but is NEVER predicted by the model.",
        "y_true": np.array([0, 1, 2, 0, 1, 2]),
        "y_pred": np.array([0, 1, 0, 0, 1, 1])
    },
    "extra_class_in_pred": {
        "comment": "Model predicts Class 3, which does NOT exist in y_true.",
        "y_true": np.array([0, 1, 2, 0, 1, 2]),
        "y_pred": np.array([0, 1, 2, 0, 3, 3])
    },
    "string_labels": {
        "comment": "Check if your library handles text/string labels like sklearn.",
        "y_true": np.array(["cat", "dog", "bird", "cat"]),
        "y_pred": np.array(["cat", "bird", "bird", "cat"])
    },
    "unseen_labels_in_test": {
        "comment": "Only a subset of global classes appear in this specific batch.",
        "y_true": np.array([1, 1, 2, 2]),
        "y_pred": np.array([1, 2, 2, 2]) # Class 0 is missing entirely from this subset
    }
}

multiclass_prob_cases = {
    # --- NORMAL CASES ---
    "multiclass_prob_normal": {
        "comment": "Standard matrix where rows sum to 1.0.",
        "y_true": np.array([0, 1, 2]),
        "y_score": np.array([
            [0.8, 0.1, 0.1], # Correct
            [0.2, 0.7, 0.1], # Correct
            [0.3, 0.4, 0.3]  # Incorrect (predicted 1, true is 2)
        ])
    },

    # --- EDGE CASES ---
    "multiclass_prob_edge_zeros": {
        "comment": "Contains absolute 0.0 and 1.0 probabilities. Hard test for Log Loss.",
        "y_true": np.array([0, 1, 5]),
        "y_score": np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0] # Completely wrong for class 1
        ])
    }
}

## Test cases

from sklearn.metrics import confusion_matrix, roc_auc_score

for idx, (testcase, data) in enumerate(binary_label_cases.items()):
    cm1 = ClassificationMetric(y_true=data["y_true"], y_pred=data["y_pred"])
    # t1 = cm1.AS()
    # t1 = cm1.PS(average="weighted", pos_label=1)
    # t1 = cm1.NPV(average="micro")
    # t1 = cm1.RS(average="macro")
    # t1 = cm1.specificity_score(average="micro")
    # t1 = cm1.F1S(average="macro")
    # t1 = cm1.F2S(average="micro")
    # t1 = cm1.FBS(average="macro", beta=1.5)
    # t1 = cm1.MCC(average="macro")
    # t1 = cm1.HML(average="macro")
    # t1 = cm1.LS(average="micro")
    # t1 = cm1.cohen_kappa_score(average="micro")
    # t1 = cm1.jaccard_similarity_index(average="macro")
    # t1 = cm1.g_mean_score(average="micro")
    # print(cm1.confusion_matrix(normalize=True))
    # print(confusion_matrix(data["y_true"], data["y_pred"]))

    # t1 = cm1.roc_auc_score(average="weighted")
    # t1 = cm1.gini_index()
    # t1 = cm1.crossentropy_loss()
    # t1 = cm1.kullback_leibler_divergence_loss()
    # t1 = cm1.hinge_loss()
    # t1 = cm1.brier_score_loss()
    t1 = cm1.KLDL()

    # t2 = 0 # f1_score(data["y_true"], data["y_pred"], average="macro")
    t2 = 0 # roc_auc_score(data["y_true"], data["y_score"])
    print(t1, t2)




