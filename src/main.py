import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from data_pipeline import DataPipeline
from dl_models import LSTMAnomalyDetector, CNN1DAnomalyDetector
from dl_trainer import train_model, set_seed
from automata_model import apply_sax, extract_sliding_patterns, build_transition_matrix
from explainability import ExplainabilityEngine
from metrics import calculate_metrics, run_wilcoxon_test
from utils import TimeSeriesDataset, add_gaussian_noise

def main():
    print("1. Konfigurasyon ve Veri Yukleme (BATADAL Ozelinde)")
    pipeline = DataPipeline("src/config.yaml")
    config = pipeline.config
    
    df, feature_cols, target_col = pipeline.load_batadal()
    train_idx, val_idx, test_idx = pipeline.split_batadal(df)
    
    # Data Leakage kuralini engelleyen PCA ve Scaler
    processed_data = pipeline.preprocess_features(df, train_idx, val_idx, test_idx, feature_cols)
    dl_data = processed_data['deep_learning']
    auto_data = processed_data['automata']
    y_test = df.loc[test_idx, target_col].values
    
    # ---------------------------------------------------------
    print("\n2. Derin Ogrenme (LSTM) Modelinin Hazirlanmasi")
    seq_len = config['automata']['default_window_size']
    train_dataset = TimeSeriesDataset(dl_data['train'], df.loc[train_idx, target_col].values, seq_len)
    val_dataset = TimeSeriesDataset(dl_data['val'], df.loc[val_idx, target_col].values, seq_len)
    test_dataset = TimeSeriesDataset(dl_data['test'], y_test, seq_len)
    
    batch_size = config['deep_learning']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    seed = config['experiment']['random_seeds'][0]
    set_seed(seed)
    
    input_size = dl_data['train'].shape[1]
    lstm_model = LSTMAnomalyDetector(
        input_size=input_size,
        hidden_size=config['deep_learning']['lstm']['hidden_size'],
        num_layers=config['deep_learning']['lstm']['num_layers'],
        dropout=config['deep_learning']['lstm']['dropout']
    )
    
    lstm_model = train_model(lstm_model, train_loader, val_loader, config, seed)
    
    # LSTM Test Edilmesi
    lstm_model.eval()
    lstm_preds = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for X_batch, _ in test_loader:
            preds = lstm_model(X_batch.to(device)).cpu().numpy()
            # If shape is [batch, 1], flatten it. If it's [batch], it's fine.
            preds = preds.flatten() if preds.ndim > 1 else preds
            lstm_preds.extend(preds)
            
    # Sequence penceresinden eksilen ilk satirlari hizalamak icin sifir ekliyoruz
    lstm_preds = [0] * (seq_len - 1) + lstm_preds
    print("LSTM Metrikleri:", calculate_metrics(y_test, lstm_preds))
    
    # ---------------------------------------------------------
    print("\n3. Beyaz Kutu: Olasiliksal Otomata ve XAI")
    alphabet_size = config['automata']['default_alphabet_size']
    
    train_symbols = apply_sax(auto_data['train'], alphabet_size=alphabet_size)
    train_patterns = extract_sliding_patterns(train_symbols, window_size=seq_len)
    automaton = build_transition_matrix([train_patterns])
    
    test_symbols = apply_sax(auto_data['test'], alphabet_size=alphabet_size)
    test_patterns = extract_sliding_patterns(test_symbols, window_size=seq_len)
    
    engine = ExplainabilityEngine(automaton=automaton, anomaly_threshold=0.01)
    os.makedirs("results", exist_ok=True)
    engine.save_json(test_patterns, "results/automata_explanation.json")
    print("JSON aciklanabilirlik raporu 'results/' klasorune kaydedildi.")
    
    explanations = engine.explain_sequence(test_patterns)
    auto_preds = [1 if e["decision"] == "anomaly" else 0 for e in explanations]
    auto_preds = [0] * (seq_len - 1) + auto_preds
    print("Automata Metrikleri:", calculate_metrics(y_test, auto_preds))
    
    # ---------------------------------------------------------
    print("\n4. Modeller Arasi Istatistiksel Analiz")
    lstm_binary = (np.array(lstm_preds) >= 0.5).astype(int)
    stat_test = run_wilcoxon_test(lstm_binary, auto_preds, y_test)
    print("Wilcoxon P-Value Test Sonuclari:", stat_test)

if __name__ == "__main__":
    main()
