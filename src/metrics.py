import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import wilcoxon

def calculate_metrics(y_true, y_pred):
    """Sınıflandırma metriklerini hesaplar."""
    # Olasılık geldiyse 0.5 eşiği ile 0 veya 1'e çevir
    if isinstance(y_pred[0], float) or np.issubdtype(type(y_pred[0]), np.floating):
        y_pred = (np.array(y_pred) >= 0.5).astype(int)
        
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

def run_wilcoxon_test(y_pred_model1, y_pred_model2, y_true):
    """
    Rubric: Modeller arası anlamlılık analizi.
    Wilcoxon Signed-Rank Test kullanarak iki modelin başarı farkının istatistiksel
    olarak anlamlı olup olmadığını (p < 0.05) test eder.
    """
    # 1: Doğru tahmin, 0: Yanlış tahmin
    scores1 = (np.array(y_pred_model1) == np.array(y_true)).astype(int)
    scores2 = (np.array(y_pred_model2) == np.array(y_true)).astype(int)
    
    # Modeller tamamen aynı tahmini yaptıysa Wilcoxon hata verebilir
    if np.array_equal(scores1, scores2):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}
        
    stat, p_value = wilcoxon(scores1, scores2)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05)
    }
