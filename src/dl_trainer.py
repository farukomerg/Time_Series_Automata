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

def train_model(model, train_loader, val_loader, config, seed):
    """
    Derin öğrenme modelleri (LSTM / CNN) için ana eğitim döngüsü (Training Loop).
    """
    set_seed(seed)
    
    epochs = config['deep_learning']['epoch_limit']
    patience = config['deep_learning']['early_stopping_patience']
    
    criterion = nn.BCELoss()
    lr = config["deep_learning"]["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"--- Eğitim Başlıyor (Seed: {seed}) ---")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device, dtype=torch.float32), y_batch.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze(-1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device, dtype=torch.float32), y_val.to(device, dtype=torch.float32)
                outputs = model(X_val).squeeze(-1)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Seed {seed}: Dogrulama kaybi {patience} tur boyunca dusmedi. Erken Durduruldu.")
            break
            
    return model
