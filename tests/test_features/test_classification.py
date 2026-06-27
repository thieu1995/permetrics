#!/usr/bin/env python
# Created by "Thieu" at 10:00, 27/07/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
import numpy as np
from permetrics import ClassificationMetric
from unittest.mock import MagicMock, patch

# =====================================================================
# FIXTURES & MOCK SETUP
# =====================================================================

@pytest.fixture
def metric():
    """Create a default instance with no parameters."""
    # Patch EPSILON for loss functions if the Evaluator superclass does not initialize automatically in the test environment.
    if not hasattr(ClassificationMetric, "EPSILON"):
        ClassificationMetric.EPSILON = 1e-15
    return ClassificationMetric()


@pytest.fixture
def mock_du():
    with patch("permetrics.classification.du") as mock:
        yield mock


@pytest.fixture
def mock_cu():
    with patch("permetrics.classification.cu") as mock:
        yield mock


# =====================================================================
# 1. TEST STARTUP & GET SUPPORT
# =====================================================================

class TestStaticAndInit:
    def test_init_with_params(self):
        m = ClassificationMetric(y_true=[1, 0], y_pred=[1, 1], custom_kw="test")
        assert m.y_true == [1, 0]
        assert m.y_pred == [1, 1]
        assert m.binary is True
        assert m.representor == "number"

    def test_get_support_all(self, capsys):
        res = ClassificationMetric.get_support("all", verbose=True)
        assert isinstance(res, dict)
        assert "F1S" in res
        assert "ROC-AUC" in res
        captured = capsys.readouterr()
        assert "Metric PS" in captured.out

    def test_get_support_single_valid(self, capsys):
        res = ClassificationMetric.get_support("MCC", verbose=True)
        assert res["range"] == "[-1, +1]"
        captured = capsys.readouterr()
        assert "Metric MCC" in captured.out

    def test_get_support_invalid(self):
        with pytest.raises(ValueError, match="doesn't support metric named: INVALID"):
            ClassificationMetric.get_support("INVALID")


# =====================================================================
# 2. TEST INPUT DATA PROCESSING (GET PROCESSED DATA)
# =====================================================================

class TestGetProcessedData:
    def test_get_processed_data_passed_directly(self, metric, mock_du):
        mock_du.format_classification_data.return_value = (np.array([1]), np.array([1]), True, "num")
        res = metric.get_processed_data([1], [1])
        mock_du.format_classification_data.assert_called_once_with([1], [1])
        assert res[2] is True

    def test_get_processed_data_from_self(self, mock_du):
        m = ClassificationMetric(y_true=[0], y_pred=[0])
        mock_du.format_classification_data.return_value = (np.array([0]), np.array([0]), True, "num")
        res = m.get_processed_data()
        mock_du.format_classification_data.assert_called_once_with([0], [0])
        assert res[0][0] == 0

    def test_get_processed_data_none_raises(self, metric):
        with pytest.raises(ValueError, match="y_true or y_pred is None"):
            metric.get_processed_data(None, [1])

    def test_get_processed_data2_passed_directly(self, metric, mock_du):
        mock_du.format_y_score.return_value = (np.array([1]), np.array([0.8]), True, "num")
        res = metric.get_processed_data2([1], [0.8])
        mock_du.format_y_score.assert_called_once_with([1], [0.8])
        assert res[1][0] == 0.8

    def test_get_processed_data2_none_raises(self, metric):
        with pytest.raises(ValueError, match="y_true or y_pred is None"):
            metric.get_processed_data2()


# =====================================================================
# 3. CORE ENGINE TEST (_AGGREGATE & MICRO STATS)
# =====================================================================

class TestAggregateEngine:
    def test_get_micro_stats(self, metric):
        # 3x3 matrix: diagonal is TP (10+15+20 = 45). Sum N = 60.
        matrix = np.array([
            [10, 2, 3],
            [1, 15, 1],
            [4, 4, 20]
        ])
        tp, fp, fn, tn = metric._get_micro_stats(matrix)
        assert tp == 45
        assert fp == 15  # 60 - 45
        assert fn == 15  # 60 - 45
        assert tn == 105  # 60*3 - (45 + 15 + 15)

    def test_aggregate_binary_exceptions(self, metric, mock_du):
        # Multiclass error but requires running binary.
        mock_du.format_classification_data.return_value = (np.array([0, 1, 2]), np.array([0, 1, 2]), [0, 1, 2], "number")
        with pytest.raises(ValueError, match="Target is multiclass"):
            metric._aggregate("precision", [0, 1, 2], [0, 1, 2], None, pos_label=1, average="binary")

        # Error: pos_label does not exist
        mock_du.format_classification_data.return_value = (np.array([0, 1]), np.array([0, 1]), [0, 1], "num")
        with pytest.raises(ValueError, match="is not a valid label"):
            metric._aggregate("precision", [0, 1], [0, 1], None, pos_label=99, average="binary")

    def test_aggregate_micro_branch(self, metric, mock_du, mock_cu):
        mock_du.format_classification_data.return_value = (np.array([0, 1]), np.array([0, 1]), [0, 1], "num")
        mock_cu.calculate_confusion_matrix.return_value = (np.array([[5, 1], [1, 5]]), {0: 0, 1: 1}, {0: 6, 1: 6})
        mock_cu.calculate_single_label_metric.return_value = {"_m": {"precision": 0.8333}}

        res = metric._aggregate("precision", [0, 1], [0, 1], None, pos_label=1, average="micro")
        assert np.isclose(res, 0.8333)

    def test_aggregate_binary_branch(self, metric, mock_du, mock_cu):
        mock_du.format_classification_data.return_value = (
        np.array(["cat", "dog"]), np.array(["cat", "dog"]), ["cat", "dog"], "string")
        mock_cu.calculate_confusion_matrix.return_value = (np.eye(2), {"cat": 0, "dog": 1}, {"cat": 1, "dog": 1})
        mock_cu.calculate_single_label_metric.return_value = {"dog": {"f1": 0.95}, "cat": {"f1": 0.85}}

        res = metric._aggregate("f1", ["cat", "dog"], ["cat", "dog"], None, pos_label="dog", average="binary")
        assert res == 0.95

    def test_aggregate_none_branch(self, metric, mock_du, mock_cu):
        mock_du.format_classification_data.return_value = (np.array([0, 1]), np.array([0, 1]), [0, 1], "number")
        mock_cu.calculate_confusion_matrix.return_value = (np.eye(2), {0: 0, 1: 1}, {0: 1, 1: 1})
        mock_cu.calculate_single_label_metric.return_value = {0: {"recall": 1.0}, 1: {"recall": 0.5}}

        res = metric._aggregate("recall", [0, 1], [0, 1], labels=[0, 1], pos_label=1, average=None)
        assert res == {0: 1.0, 1: 0.5}

    def test_aggregate_macro_and_weighted_branch(self, metric, mock_du, mock_cu):
        mock_du.format_classification_data.return_value = (np.array([0, 1]), np.array([0, 1]), [0, 1], "number")
        mock_cu.calculate_confusion_matrix.return_value = (np.eye(2), {0: 0, 1: 1}, {0: 10, 1: 30})
        mock_cu.calculate_single_label_metric.return_value = {
            0: {"acc": 0.4, "n_true": 10},
            1: {"acc": 0.8, "n_true": 30}
        }

        # Macro = (0.4 + 0.8) / 2 = 0.6
        res_macro = metric._aggregate("acc", [0, 1], [0, 1], labels=None, pos_label=1, average="macro")
        assert np.isclose(res_macro, 0.6)

        # Weighted = (0.4*10 + 0.8*30) / 40 = 28 / 40 = 0.7
        res_weighted = metric._aggregate("acc", [0, 1], [0, 1], labels=None, pos_label=1, average="weighted")
        assert np.isclose(res_weighted, 0.7)

    def test_aggregate_invalid_labels_or_average(self, metric, mock_du, mock_cu):
        mock_du.format_classification_data.return_value = (np.array([0, 1]), np.array([0, 1]), [0, 1], "number")
        mock_cu.calculate_confusion_matrix.return_value = (np.eye(2), {0: 0, 1: 1}, {0: 1, 1: 1})
        mock_cu.calculate_single_label_metric.return_value = {}

        with pytest.raises(ValueError, match="Specified labels do not exist"):
            metric._aggregate("acc", [0, 1], [0, 1], labels=[999], pos_label=1, average="macro")

        with pytest.raises(ValueError, match="Unsupported average setting"):
            metric._aggregate("acc", [0, 1], [0, 1], labels=None, pos_label=1, average="invalid_avg")


# =====================================================================
# 4. TEST GROUP OF SINGLE-LABEL & ALIASES APIS
# =====================================================================

@pytest.mark.parametrize("method_name, alias_names, metric_key", [
    ("precision_score", ["PS"], "precision"),
    ("negative_predictive_value", ["NPV"], "negative_predictive_value"),
    ("specificity_score", ["SS"], "specificity"),
    ("recall_score", ["RS"], "recall"),
    ("accuracy_score", ["AS"], "accuracy"),
    ("f1_score", ["F1S"], "f1"),
    ("f2_score", ["F2S"], "f2"),
    ("fbeta_score", ["FBS"], "fbeta"),
    ("matthews_correlation_coefficient", ["MCC"], "mcc"),
    ("hamming_loss", ["HML"], "hamming_loss"),
    ("lift_score", ["LS"], "lift_score"),
    ("cohen_kappa_score", ["CKS"], "kappa_score"),
    ("jaccard_similarity_index", ["JSI", "JSC", "jaccard_similarity_coefficient"], "jaccard_similarities"),
    ("g_mean_score", ["GMS"], "g_mean")
])
def single_label_apis(metric, method_name, alias_names, metric_key):
    """Verify that all API calls correctly use the keyword in the _aggregate function and that the aliases work."""
    with patch.object(metric, "_aggregate", return_value=0.99) as mock_agg:
        func = getattr(metric, method_name)
        res = func([1], [1], average="macro")
        assert res == 0.99
        mock_agg.assert_called_once()
        assert mock_agg.call_args[0][0] == metric_key

        # Check Aliases
        for alias in alias_names:
            alias_func = getattr(metric, alias)
            assert alias_func([1], [1]) == 0.99


# =====================================================================
# 5. TEST GROUP ROC, GINI, AND LOSS FUNCTIONS (PROB / SCORE / MIX)
# =====================================================================

class TestROCAndLosses:
    def test_confusion_matrix_alias(self, metric, mock_du, mock_cu):
        mock_du.format_classification_data.return_value = (np.array([1]), np.array([1]), True, "num")
        mock_cu.calculate_confusion_matrix.return_value = (np.array([[10]]), None, None)

        assert metric.confusion_matrix([1], [1])[0][0] == 10
        assert metric.CM([1], [1])[0][0] == 10  # Alias test

    def test_roc_auc_single_class_raises(self, metric, mock_du):
        mock_du.format_y_score.return_value = (np.array([1, 1, 1]), np.array([0.2, 0.5, 0.9]), False, "num")
        with pytest.raises(ValueError, match="Only one class present in y_true"):
            metric.roc_auc_score([1, 1, 1], [0.2, 0.5, 0.9])

    def test_roc_auc_binary_1d_score(self, metric, mock_du, mock_cu):
        mock_du.format_y_score.return_value = (np.array([0, 1]), np.array([0.1, 0.9]), True, "num")
        mock_cu.calculate_roc_curve.return_value = (np.array([0, 1]), np.array([0, 1]), None)

        res = metric.roc_auc_score([0, 1], [0.1, 0.9])
        assert np.isclose(res, 0.5)  # Diện tích tam giác nửa ô vuông chuẩn hóa

    def test_roc_auc_binary_2d_score(self, metric, mock_du, mock_cu):
        mock_du.format_y_score.return_value = (
            np.array([0, 1]),
            np.array([[0.9, 0.1], [0.2, 0.8]]),
            True, "num"
        )
        mock_cu.calculate_roc_curve.return_value = (np.array([0, 1]), np.array([0, 1]), None)

        res = metric.roc_auc_score([0, 1], [[0.9, 0.1], [0.2, 0.8]])
        assert isinstance(res, float)
        assert metric.ROC == metric.roc_auc_score  # Alias check

    def test_roc_auc_binary_invalid_shape_raises(self, metric, mock_du):
        mock_du.format_y_score.return_value = (np.array([0, 1]), np.array([[0.1, 0.2, 0.7]]), True, "num")
        with pytest.raises(ValueError, match="Target is binary but y_score has 3 columns"):
            metric.roc_auc_score([0, 1], [[0.1, 0.2, 0.7]])

    def test_roc_auc_multiclass_modes(self, metric, mock_du, mock_cu):
        y_t = np.array([0, 1, 2])
        y_s = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
        mock_du.format_y_score.return_value = (y_t, y_s, False, "num")
        mock_cu.calculate_roc_curve.return_value = (np.array([0, 1]), np.array([0, 1]), None)
        mock_cu.calculate_class_support.return_value = np.array([10, 10, 20])

        assert isinstance(metric.roc_auc_score(y_t, y_s, average="macro"), float)
        assert isinstance(metric.roc_auc_score(y_t, y_s, average="weighted"), float)

        res_none = metric.roc_auc_score(y_t, y_s, average=None)
        assert isinstance(res_none, dict)
        assert len(res_none) == 3

    def test_gini_index(self, metric):
        with patch.object(metric, "roc_auc_score", return_value=0.75):
            assert np.isclose(metric.gini_index([0, 1], [0.2, 0.8]), 0.5)   # 2 * 0.75 - 1
            assert metric.GINI == metric.gini_index

        with patch.object(metric, "roc_auc_score", return_value={"classA": 0.8}):
            assert metric.gini_index([0, 1], [0.2, 0.8]) == pytest.approx({"classA": 0.6})

    def test_loss_functions_normal_and_mix(self, metric, mock_du):
        # Set the input Mix: y_true as a string/number formatted as an int label [0, 1]
        y_true = np.array([0., 1.])
        y_pred = np.array([[0.8, 0.2], [0.1, 0.9]])
        mock_du.format_y_score.return_value = (y_true, y_pred, True, "mix")

        # 1. Brier Score Loss
        bsl = metric.brier_score_loss([0, 1], y_pred)
        assert bsl >= 0.0
        assert metric.BSL == metric.brier_score_loss

        # 2. Cross Entropy Loss (with clip traps having extreme probabilities of 0.0 and 1.0)
        y_pred_extreme = np.array([[1.0, 0.0], [0.0, 1.0]])
        mock_du.format_y_score.return_value = (y_true, y_pred_extreme, True, "num")
        cel = metric.crossentropy_loss([0, 1], y_pred_extreme)
        assert cel >= 0.0
        assert metric.CEL == metric.crossentropy_loss

        # 3. Hinge Loss
        mock_du.format_y_score.return_value = (y_true, y_pred, True, "num")
        hl = metric.hinge_loss([0, 1], y_pred)
        assert hl >= 0.0
        assert metric.HGL == metric.hinge_loss

        # 4. Kullback-Leibler Divergence Loss
        kldl = metric.kullback_leibler_divergence_loss([0, 1], y_pred)
        assert kldl >= 0.0
        assert metric.KLDL == metric.kullback_leibler_divergence_loss


# =====================================================================
# 1. MASTER DATASET ZOO (Includes all Normal and Edge Cases of the input)
# =====================================================================

@pytest.fixture(scope="module")
def clf_data_zoo():
    """
    The dataset is used for 14 Single-Label Metrics & Confusion Matrixes.
    Including: Number, String, Mix, 1D, 2D One-hot, Probabilities, Perfect, All-Wrong.
    Returns list of tuple: (instance, is_binary_problem)
    """
    zoo = []

    # 1. Normal: Binary 1D - Number
    zoo.append((ClassificationMetric([0, 1, 0, 1, 0, 1], [0, 1, 0, 0, 0, 1]), True))

    # 2. Normal: Binary 1D - String
    zoo.append((ClassificationMetric(["cat", "dog", "cat", "dog"], ["cat", "dog", "dog", "dog"]), True))

    # 3. Normal: Multiclass 1D - Number
    zoo.append((ClassificationMetric([0, 1, 2, 0, 2], [0, 2, 2, 0, 1]), False))

    # 4. Normal: Multiclass 1D - String
    zoo.append((ClassificationMetric(["A", "B", "C", "A"], ["A", "C", "C", "B"]), False))

    # 5. Edge: Mix Types
    zoo.append((ClassificationMetric([0, 1, 0], [0.0, 1.0, 0.0]), True))

    # 6. Edge: 2D One-Hot (y_true) vs 2D Probabilities (y_pred)
    y_t_oh = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]])
    y_p_prob = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.2, 0.3, 0.5], [0.3, 0.6, 0.1], [0.1, 0.2, 0.7]])
    zoo.append((ClassificationMetric(y_t_oh, y_p_prob), False))

    # 7. Edge: 1D Class Labels vs 2D Probabilities
    zoo.append((ClassificationMetric(np.array([1, 0, 2, 1, 2]), y_p_prob), False))

    # 8. Edge: Perfect Prediction (Perfect prediction 100%)
    zoo.append((ClassificationMetric(["X", "Y", "Z"], ["X", "Y", "Z"]), False))

    # 9. Edge: All Wrong (0%)
    zoo.append((ClassificationMetric([1, 1, 1], [0, 0, 0]), True))

    # 10. Edge: Blind Model (Extremely Imbalanced - The model never guessed class 2.)
    zoo.append((ClassificationMetric([0, 1, 2, 2], [0, 1, 1, 1]), False))

    return zoo


@pytest.fixture(scope="module")
def prob_data_zoo():
    """
    The dataset MUST be in numerical format (Scores/Probabilities).
    For this group only: ROC-AUC, GINI, CrossEntropy, Hinge, KL-Divergence, Brier.
    """
    zoo = []

    # 1. Binary: 1D True vs 1D Score
    zoo.append(ClassificationMetric([0, 1, 0, 1], [0.1, 0.85, 0.2, 0.99]))

    # 2. Binary: 1D True vs 2D Probabilities (Sklearn standard)
    zoo.append(ClassificationMetric([0, 1], [[0.9, 0.1], [0.2, 0.8]]))

    # 3. Multiclass: 1D True vs 2D Probabilities
    y_t = np.array([0, 1, 2])
    y_p = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])
    zoo.append(ClassificationMetric(y_t, y_p))

    # 4. Multiclass: 2D One-Hot True vs 2D Probabilities
    y_t_oh = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    zoo.append(ClassificationMetric(y_t_oh, y_p))

    # 5. Extreme Edge: Contains exactly 0.0 and 1.0 probabilities (CrossEntropy Log(0) infinite trap test)
    zoo.append(ClassificationMetric([0, 1], [[1.0, 0.0], [0.0, 1.0]]))

    return zoo


# =====================================================================
# 2. SINGLE-LABEL METRICS GROUP TEST (14 APIs)
# =====================================================================

SINGLE_LABEL_METRICS = ["PS", "NPV", "RS", "F1S", "F2S", "FBS", "SS", "MCC", "HML", "LS", "CKS", "JSI", "GMS"]


@pytest.mark.parametrize("metric_alias", SINGLE_LABEL_METRICS)
def test_single_label_multiclass_compatible(clf_data_zoo, metric_alias):
    """Test 4 average modes supporting multiclass across the entire dataset."""
    avg_paras = [None, "macro", "micro", "weighted"]
    expected_types = (dict, float, float, float)

    for cm, _ in clf_data_zoo:
        func = getattr(cm, metric_alias)
        for idx, avg in enumerate(avg_paras):
            res = func(average=avg)
            assert isinstance(res, expected_types[idx]), f"Error from {metric_alias} with average={avg}"


@pytest.mark.parametrize("metric_alias", SINGLE_LABEL_METRICS)
def test_single_label_binary_mode_only(clf_data_zoo, metric_alias):
    """The average='binary' mode is only allowed to run on binary problems."""
    for cm, is_binary in clf_data_zoo:
        if is_binary:
            y_true, y_pred, unique_classes, _ = cm.get_processed_data()
            current_pos_label = unique_classes[0] if len(unique_classes) > 0 else 1
            res = getattr(cm, metric_alias)(average="binary", pos_label=current_pos_label)
            assert isinstance(res, float)


def test_accuracy_score_multiclass(clf_data_zoo):
    for cm, _ in clf_data_zoo:
        assert isinstance(cm.AS(normalize=True), float)
        assert isinstance(cm.AS(normalize=False), int)


def test_confusion_matrix(clf_data_zoo):
    for cm, _ in clf_data_zoo:
        matrix = cm.CM()
        assert isinstance(matrix, np.ndarray)
        assert matrix.ndim == 2


# =====================================================================
# 3. TEST GROUPS: ROC, GINI, AND LOSS FUNCTIONS
# =====================================================================

@pytest.mark.parametrize("metric_alias", ["ROC", "GINI"])
def test_roc_and_gini_engine(prob_data_zoo, metric_alias):
    avg_paras = ["macro", "weighted", None]

    for cm in prob_data_zoo:
        func = getattr(cm, metric_alias)
        for avg in avg_paras:
            res = func(average=avg)
            assert isinstance(res, (float, dict))


@pytest.mark.parametrize("loss_alias", ["CEL", "HGL", "KLDL", "BSL"])
def test_loss_functions(prob_data_zoo, loss_alias):
    for cm in prob_data_zoo:
        loss_func = getattr(cm, loss_alias)
        val = loss_func()
        assert isinstance(val, float)
        assert val >= 0.0, f"Loss function {loss_alias} returns negative value: {val}"


# ==========================================================================
# 4. TEST NEGATIVE EDGE CASES (Exception traps require raising a ValueError)
# ==========================================================================

def test_strict_boundary_exceptions():
    # 1. The data is multiclass but intentionally forced to have average equal to 'binary'.
    cm_multi = ClassificationMetric([0, 1, 2], [0, 1, 2])
    with pytest.raises(ValueError, match="Target is multiclass"):
        cm_multi.PS(average="binary")

    # 2. The set y_true has only one class (ROC-AUC cannot be defined).
    cm_one_class = ClassificationMetric([1, 1, 1, 1], [0.1, 0.5, 0.8, 0.2])
    with pytest.raises(ValueError, match="Only one class present in y_true"):
        cm_one_class.ROC()

    # 3. Do not pass data during either the initialization or function calls.
    cm_empty = ClassificationMetric()
    with pytest.raises(ValueError, match="y_true or y_pred is None"):
        cm_empty.get_processed_data()

    # 4. Calling a metric that doesn't exist.
    with pytest.raises(ValueError, match="doesn't support metric named"):
        ClassificationMetric.get_support("NON_EXISTENT_METRIC")
