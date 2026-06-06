import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import wilcoxon


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Sınıflandırma metriklerini hesaplar."""
    if len(y_pred) > 0 and (
        isinstance(y_pred[0], float) or np.issubdtype(type(y_pred[0]), np.floating)
    ):
        y_pred = (np.array(y_pred) >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def run_wilcoxon_test(y_pred_model1, y_pred_model2, y_true):
    """Örnek bazlı doğru/yanlış tahmin skorları üzerinde Wilcoxon testi."""
    scores1 = (np.array(y_pred_model1) == np.array(y_true)).astype(int)
    scores2 = (np.array(y_pred_model2) == np.array(y_true)).astype(int)
    if np.array_equal(scores1, scores2):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
    stat, p_value = wilcoxon(scores1, scores2)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def run_wilcoxon_paired_f1(scores_a, scores_b):
    """Seed bazlı eşleşmiş F1 skorları üzerinde Wilcoxon testi."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) != len(b) or len(a) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
    if np.array_equal(a, b):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
    stat, p_value = wilcoxon(a, b)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }
