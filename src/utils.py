import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    """Derin öğrenme modelleri (LSTM, 1D-CNN) için kayan pencere (sliding window) veri seti."""
    def __init__(self, X, y, sequence_length=4):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        
    def __len__(self):
        return max(0, len(self.X) - self.sequence_length + 1)
        
    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.sequence_length]
        # Hedef değişken (anomaly flag) genellikle pencerenin son elemanına aittir
        y_val = self.y[idx + self.sequence_length - 1]
        
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

def add_gaussian_noise(X, mean, std):
    """
    Rubric kuralı: Sistemin dayanıklılığını test etmek için 
    veriye Gauss gürültüsü (Gaussian Noise) ekler.
    """
    noise = np.random.normal(mean, std, size=X.shape)
    return X + noise
