import torch
import torch.nn as nn
import numpy as np

def set_seed(seed):
    """
    Sonuçların tekrarlanabilir (reproducible) olması için tüm rastgelelikleri sabitler.
    Rubric'te istenen 5 farklı seed (42, 123, 2026, 7, 999) burada kullanılacaktır.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    """
    Eğitim sırasında validation (doğrulama) kaybı iyileşmeyi kestiğinde eğitimi erken durdurur.
    Rubric: validation loss (patience= 5) kuralını uygular.
    """
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
