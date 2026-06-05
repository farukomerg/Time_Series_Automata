import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data_pipeline import DataPipeline
from dl_models import LSTMAnomalyDetector, CNN1DAnomalyDetector
from dl_trainer import train_model, set_seed
from automata_model import apply_sax, extract_sliding_patterns, build_transition_matrix
from explainability import ExplainabilityEngine
from metrics import calculate_metrics, run_wilcoxon_test
from utils import TimeSeriesDataset, add_gaussian_noise
from visualization import plot_confusion_matrix, plot_roc_curve, plot_automata_state_diagram, plot_transition_heatmap

def evaluate_model_predictions(model, test_loader, seq_len, device):
    """Derin öğrenme modelinden olasılıkları ve binary tahminleri toplar, boyutu hizalar."""
    model.eval()
    probs = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            preds = model(X_batch.to(device)).cpu().numpy().flatten()
            probs.extend(preds)
            
    # Pencerelemeden dolayı eksilen ilk satırları hizalamak için 0.0 ekliyoruz (Boyut hatası çözümü)
    probs = [0.0] * (seq_len - 1) + probs
    binary = (np.array(probs) >= 0.5).astype(int)
    return probs, binary

def run_experiment_pipeline():
    print("=== Gelişmiş Zaman Serisi Otomata ve DL Değerlendirme Sistemi ===")
    pipeline = DataPipeline()
    config = pipeline.config
    seq_len = config['automata']['default_window_size']
    alphabet_size = config['automata']['default_alphabet_size']
    batch_size = config['deep_learning']['batch_size']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # İster 1: Her iki veri seti döngüsü (BATADAL ve SKAB)
    datasets = ["BATADAL", "SKAB"]
    
    for dataset_name in datasets:
        print(f"\n==================== VERİ SETİ: {dataset_name} ====================")
        
        # Veri setine göre yükleme fonksiyonunu dinamik seç
        if dataset_name == "BATADAL":
            df, feature_cols, target_col = pipeline.load_batadal()
            # Tek bir büyük split split_batadal'dan gelir
            splits = [pipeline.split_batadal(df)]
        else:
            df, feature_cols, target_col = pipeline.load_skab()
            # SKAB için GroupKFold generator adımları
            splits = list(pipeline.get_skab_splits(df))
            
        for scenario in config['experiment']['scenarios']:
            print(f"\n--- Senaryo: {scenario} ---")
            
            # Sonuçları biriktirmek için tablolar (Mean +- Std hesaplaması için)
            lstm_f1_scores = []
            cnn_f1_scores = []
            auto_f1_scores = []
            
            for seed in config['experiment']['random_seeds']:
                set_seed(seed)
                
                
                # Split döngüsü (BATADAL için 1 adet, SKAB için fold yapısı)
                for split_idx, split_info in enumerate(splits):
                    # BATADAL 3 değer döner (train, val, test), SKAB 2 değer döner (train, test)
                    if len(split_info) == 3:
                        train_idx, val_idx, test_idx = split_info
                    else:
                        train_val_idx, test_idx = split_info
                        # SKAB için train verisinin son %20'sini validation (doğrulama) olarak ayırıyoruz
                        split_point = int(len(train_val_idx) * 0.8)
                        train_idx = train_val_idx[:split_point]
                        val_idx = train_val_idx[split_point:]
                    
                    # Veri sızıntısını engelleyen merkezi pipeline ön işlemesi
                    processed_data = pipeline.preprocess_features(df, train_idx, val_idx, test_idx, feature_cols)
                    dl_data = processed_data['deep_learning']
                    auto_data = processed_data['automata']
                    
                    # Test seti verilerini senaryoya göre manipüle et (Gürültü İsteri)
                    X_test_dl = dl_data['test'].copy()
                    X_test_auto = auto_data['test'].copy()
                    if scenario == "gaussian_noise":
                        X_test_dl = add_gaussian_noise(X_test_dl, std=0.1)
                        X_test_auto = add_gaussian_noise(X_test_auto, std=0.1)
                    elif scenario == "unseen_data":
                        # Unseen veride test örüntülerine yapay kaydırma eklenerek 'unseen' durum tetiklenir
                        X_test_auto = X_test_auto * 1.5
                        
                    y_test = df.loc[test_idx, target_col].values
                    
                    # Dinamik Klasörleme Mimarisi (Windows Uyumlu Dosya İsimleri ile)
                    split_label = f"fold_{split_idx}" if dataset_name == "SKAB" else "full"
                    current_results_dir = os.path.join(
                        project_root, "results", dataset_name, scenario, f"seed_{seed}", split_label
                    )
                    os.makedirs(current_results_dir, exist_ok=True)
                    
                    # PyTorch Veri Yükleyicileri
                    train_dataset = TimeSeriesDataset(dl_data['train'], df.loc[train_idx, target_col].values, seq_len)
                    val_dataset = TimeSeriesDataset(dl_data['val'], df.loc[val_idx, target_col].values, seq_len)
                    test_dataset = TimeSeriesDataset(X_test_dl, y_test, seq_len)
                    
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                    
                    # 1. MODEL: LSTM Düzenleme ve Eğitimi
                    lstm_model = LSTMAnomalyDetector(
                        input_size=dl_data['train'].shape[1],
                        hidden_size=config['deep_learning']['lstm']['hidden_size'],
                        num_layers=config['deep_learning']['lstm']['num_layers'],
                        dropout=config['deep_learning']['lstm']['dropout']
                    )
                    lstm_model = train_model(lstm_model, train_loader, val_loader, config, seed)
                    lstm_probs, lstm_binary = evaluate_model_predictions(lstm_model, test_loader, seq_len, device)
                    
                    # 2. MODEL: CNN-1D Düzenleme ve Eğitimi (Eksik Rubrik Maddesi Tamamlandı)
                    cnn_model = CNN1DAnomalyDetector(
                        input_size=dl_data['train'].shape[1],
                        filters=config['deep_learning']['cnn_1d']['filters'],
                        kernel_size=config['deep_learning']['cnn_1d']['kernel_size']
                    )
                    cnn_model = train_model(cnn_model, train_loader, val_loader, config, seed)
                    cnn_probs, cnn_binary = evaluate_model_predictions(cnn_model, test_loader, seq_len, device)
                    
                    # 3. MODEL: Beyaz Kutu Olasılıksal Otomata Yapısı (Senin Modülün)
                    train_symbols = apply_sax(auto_data['train'], alphabet_size=alphabet_size)
                    train_patterns = extract_sliding_patterns(train_symbols, window_size=seq_len)
                    automaton = build_transition_matrix([train_patterns])
                    
                    test_symbols = apply_sax(X_test_auto, alphabet_size=alphabet_size)
                    test_patterns = extract_sliding_patterns(test_symbols, window_size=seq_len)
                    
                    engine = ExplainabilityEngine(automaton=automaton, anomaly_threshold=0.01)
                    engine.save_json(test_patterns, os.path.join(current_results_dir, "automata_explanation.json"))
                    
                    explanations = engine.explain_sequence(test_patterns)
                    auto_preds = [1 if e["decision"] == "anomaly" else 0 for e in explanations]
                    auto_preds = [0] * (seq_len - 1) + auto_preds
                    
                    # İstatistiksel Metriklerin Hesaplanıp Listelere Eklenmesi
                    lstm_metrics = calculate_metrics(y_test, lstm_binary)
                    cnn_metrics = calculate_metrics(y_test, cnn_binary)
                    auto_metrics = calculate_metrics(y_test, auto_preds)
                    
                    lstm_f1_scores.append(lstm_metrics['f1'])
                    cnn_f1_scores.append(cnn_metrics['f1'])
                    auto_f1_scores.append(auto_metrics['f1'])
                    
                    # Görsel Çıktıların Windows Karakter Kurallarına Göre Kaydedilmesi
                    plot_confusion_matrix(y_test, lstm_binary, f"LSTM-{dataset_name}-{scenario}", save_dir=current_results_dir)
                    plot_confusion_matrix(y_test, cnn_binary, f"CNN1D-{dataset_name}-{scenario}", save_dir=current_results_dir)
                    plot_confusion_matrix(y_test, auto_preds, f"Automata-{dataset_name}-{scenario}", save_dir=current_results_dir)
                    
                    plot_roc_curve(y_test, lstm_probs, f"LSTM-{dataset_name}-{scenario}", save_dir=current_results_dir)
                    plot_roc_curve(y_test, cnn_probs, f"CNN1D-{dataset_name}-{scenario}", save_dir=current_results_dir)
                    
                    plot_automata_state_diagram(automaton.to_dataframe(), save_dir=current_results_dir)
                    plot_transition_heatmap(automaton.to_dataframe(), save_dir=current_results_dir)
            
            # Her senaryo sonunda akademik "Mean +- Std" rapor tablosunun terminale dökülmesi
            print(f"\n>>> BAŞARI RAPORU ({dataset_name} - {scenario}) <<<")
            print(f"LSTM Ortalama F1-Skoru : {np.mean(lstm_f1_scores):.4f} ± {np.std(lstm_f1_scores):.4f}")
            print(f"1D-CNN Ortalama F1-Skoru: {np.mean(cnn_f1_scores):.4f} ± {np.std(cnn_f1_scores):.4f}")
            print(f"Otomata Ortalama F1-Skoru: {np.mean(auto_f1_scores):.4f} ± {np.std(auto_f1_scores):.4f}")

if __name__ == "__main__":
    run_experiment_pipeline()