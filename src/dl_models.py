import torch
import torch.nn as nn

class LSTMAnomalyDetector(nn.Module):
    """
    Zaman serisi anomali tespiti için PyTorch tabanlı LSTM modeli.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMAnomalyDetector, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x boyutu: (batch_size, sequence_length, num_features)
        lstm_out, (hn, cn) = self.lstm(x)
        # Yalnızca en son zaman adımının (t=sequence_length) çıktısını alıyoruz
        last_out = lstm_out[:, -1, :] 
        # Anomali olasılığı (0 veya 1 aralığında) için Sigmoid fonksiyonu kullanıyoruz
        return torch.sigmoid(self.fc(last_out))


class CNN1DAnomalyDetector(nn.Module):
    """
    Zaman serisi anomali tespiti için 1 Boyutlu CNN modeli.
    """
    def __init__(self, input_size, filters, kernel_size):
        super(CNN1DAnomalyDetector, self).__init__()
        # Conv1d girdi olarak (batch_size, num_features, sequence_length) bekler.
        self.conv1 = nn.Conv1d(
            in_channels=input_size, 
            out_channels=filters, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2  # Zaman boyutunu korumak için padding
        )
        self.relu = nn.ReLU()
        # Zaman boyutunu (sequence_length) 1'e indirgeyerek özellikleri özetler
        self.pool = nn.AdaptiveAvgPool1d(1) 
        self.fc = nn.Linear(filters, 1)
        
    def forward(self, x):
        # x başlangıç boyutu: (batch_size, sequence_length, input_size)
        # Conv1D için boyutları yer değiştiriyoruz: (batch_size, input_size, sequence_length)
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.relu(x)
        
        x = self.pool(x).squeeze(-1) # Boyut: (batch_size, filters)
        return torch.sigmoid(self.fc(x))
