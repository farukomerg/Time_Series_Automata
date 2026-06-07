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
    Ek olarak: En iyi ağırlıkları kaydeder ve geri yüklemeyi sağlar.
    """
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state_dict = None
        
    def __call__(self, val_loss, model):
        import copy
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state_dict = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.counter = 0

def train_model(model, train_loader, val_loader, config, seed):
    """
    Derin öğrenme modelleri (LSTM / CNN) için ana eğitim döngüsü (Training Loop).
    """
    set_seed(seed)
    
    epochs = config['deep_learning']['epoch_limit']
    patience = config['deep_learning']['early_stopping_patience']
    
    # Calculate pos_weight dynamically from the training loader labels (handling class imbalance)
    # Cap to 10.0 to prevent validation loss explosion under extreme imbalance
    all_y = []
    for _, y in train_loader:
        all_y.extend(y.numpy())
    all_y = np.array(all_y)
    pos_count = np.sum(all_y == 1.0)
    neg_count = np.sum(all_y == 0.0)
    if pos_count > 0:
        pos_weight = min(neg_count / pos_count, 10.0)
    else:
        pos_weight = 1.0
        
    lr = config["deep_learning"]["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
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
            
            # Apply weights: pos_weight for positive class, 1.0 for negative class
            weight = torch.ones_like(y_batch)
            weight[y_batch == 1.0] = pos_weight
            criterion = nn.BCELoss(weight=weight)
            
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
                
                weight = torch.ones_like(y_val)
                weight[y_val == 1.0] = pos_weight
                criterion = nn.BCELoss(weight=weight)
                
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Seed {seed}: Dogrulama kaybi {patience} tur boyunca dusmedi. Erken Durduruldu.")
            break
            
    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)
        
    return model
