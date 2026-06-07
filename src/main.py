import os
import json
import csv
<<<<<<< HEAD
=======
import time
>>>>>>> c95c420aef4ed2407790f8f0e0e83a6cfcd7ce9f
import torch
import numpy as np
from torch.utils.data import DataLoader

from data_pipeline import DataPipeline
from dl_models import LSTMAnomalyDetector, CNN1DAnomalyDetector
from dl_trainer import train_model, set_seed
from automata_model import (
    apply_paa,
    apply_sax,
    extract_sliding_patterns,
    build_transition_matrix,
)
from explainability import ExplainabilityEngine
<<<<<<< HEAD
from metrics import calculate_metrics, run_wilcoxon_test, run_wilcoxon_paired_f1
=======
from metrics import calculate_metrics, run_wilcoxon_paired_f1
>>>>>>> c95c420aef4ed2407790f8f0e0e83a6cfcd7ce9f
from utils import TimeSeriesDataset, add_gaussian_noise
from visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_automata_state_diagram,
    plot_transition_heatmap,
    plot_parameter_sensitivity,
)


def evaluate_model_predictions(model, test_loader, seq_len, device, threshold):
    model.eval()
    probs = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            preds = model(X_batch.to(device)).cpu().numpy().flatten()
            probs.extend(preds)
    probs = [0.0] * (seq_len - 1) + probs
    binary = (np.array(probs) >= threshold).astype(int)
    return probs, binary


<<<<<<< HEAD
=======
def evaluate_val_predictions(model, val_loader, seq_len, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            preds = model(X_batch.to(device)).cpu().numpy().flatten()
            probs.extend(preds)
    probs = [0.0] * (seq_len - 1) + probs
    return probs


def optimize_threshold(probs, y_true):
    best_thresh = 0.5
    best_f1 = -1.0
    for thresh in np.linspace(0.01, 0.99, 99):
        binary = (np.array(probs) >= thresh).astype(int)
        m = calculate_metrics(y_true, binary)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = thresh
    return best_thresh


>>>>>>> c95c420aef4ed2407790f8f0e0e83a6cfcd7ce9f
def pc1_to_patterns(pc1_series, paa_window, alphabet_size, pattern_window):
    paa_values = apply_paa(pc1_series, window_size=paa_window)
    symbols = apply_sax(paa_values, alphabet_size=alphabet_size)
    return extract_sliding_patterns(symbols, window_size=pattern_window)


def compute_automata_stats(automaton):
    state_count = len(automaton.states)
    if state_count == 0:
        return 0, 0.0
    matrix = automaton.to_dataframe()
    nonzero = int((matrix.values > 0).sum())
    density = nonzero / (state_count * state_count)
    return state_count, float(density)


def _log_line(log_path, message):
    print(message)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def _append_metric_row(
    rows,
    dataset,
    scenario,
    window_size,
    alphabet_size,
    seed,
    split,
    model,
    metrics,
    state_count,
    transition_density,
<<<<<<< HEAD
=======
    training_time=0.0,
    inference_time=0.0,
    mapping_accuracy="N/A",
    detection_rate="N/A",
>>>>>>> c95c420aef4ed2407790f8f0e0e83a6cfcd7ce9f
):
    rows.append(
        {
            "dataset": dataset,
            "scenario": scenario,
            "window_size": window_size,
            "alphabet_size": alphabet_size,
            "seed": seed,
            "split": split,
            "model": model,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "state_count": state_count,
            "transition_density": transition_density,
<<<<<<< HEAD
=======
            "training_time": training_time,
            "inference_time": inference_time,
            "mapping_accuracy": mapping_accuracy,
            "detection_rate": detection_rate,
>>>>>>> c95c420aef4ed2407790f8f0e0e83a6cfcd7ce9f
        }
    )


def _save_experiment_logs(rows, path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


<<<<<<< HEAD
=======
def run_cross_dataset_automata(pipeline, config):
    print("\n===== ÇAPRAZ VERİ SETİ (CROSS-DATASET) ANALİZİ (Otomata) =====")
    window_size = config["automata"]["default_window_size"]
    alphabet_size = config["automata"]["default_alphabet_size"]
    paa_window = config["automata"]["paa_window_size"]
    anomaly_threshold = config["automata"]["anomaly_threshold"]
    smoothing_k = config["automata"].get("smoothing_k", 0.1)

    df_skab, skab_feats, skab_target = pipeline.load_skab()
    df_bat, bat_feats, bat_target = pipeline.load_batadal()

    skab_splits = list(pipeline.get_skab_splits(df_skab))
    train_val_idx_skab, test_idx_skab = skab_splits[0]
    split_point = int(len(train_val_idx_skab) * config["experiment"]["skab_train_val_split"])
    train_idx_skab = train_val_idx_skab[:split_point]
    val_idx_skab = train_val_idx_skab[split_point:]

    train_idx_bat, val_idx_bat, test_idx_bat = pipeline.split_batadal(df_bat)

    p_skab = pipeline.preprocess_features(df_skab, train_idx_skab, val_idx_skab, test_idx_skab, skab_feats)
    p_bat = pipeline.preprocess_features(df_bat, train_idx_bat, val_idx_bat, test_idx_bat, bat_feats)

    train_pat_skab = pc1_to_patterns(p_skab["automata"]["train"], paa_window, alphabet_size, window_size)
    test_pat_skab = pc1_to_patterns(p_skab["automata"]["test"], paa_window, alphabet_size, window_size)

    train_pat_bat = pc1_to_patterns(p_bat["automata"]["train"], paa_window, alphabet_size, window_size)
    test_pat_bat = pc1_to_patterns(p_bat["automata"]["test"], paa_window, alphabet_size, window_size)

    auto_skab = build_transition_matrix([train_pat_skab], smoothing_k=smoothing_k)
    auto_bat = build_transition_matrix([train_pat_bat], smoothing_k=smoothing_k)

    # Cross 1: Train SKAB -> Test BATADAL
    engine_skab_to_bat = ExplainabilityEngine(
        automaton=auto_skab,
        training_vocabulary=set(train_pat_skab),
        anomaly_threshold=anomaly_threshold,
        window_size=window_size
    )
    explanations_1 = engine_skab_to_bat.explain_sequence(test_pat_bat)
    preds_1 = [1 if e["decision"] == "anomaly" else 0 for e in explanations_1]
    preds_1_padded = [0] * (window_size - 1) + preds_1
    preds_1_upsampled = np.repeat(preds_1_padded, paa_window)
    y_test_bat = df_bat.loc[test_idx_bat, bat_target].values
    if len(preds_1_upsampled) < len(y_test_bat):
        preds_1_upsampled = np.concatenate([preds_1_upsampled, np.zeros(len(y_test_bat) - len(preds_1_upsampled))])
    else:
        preds_1_upsampled = preds_1_upsampled[:len(y_test_bat)]
    m_1 = calculate_metrics(y_test_bat, preds_1_upsampled)

    # Cross 2: Train BATADAL -> Test SKAB
    engine_bat_to_skab = ExplainabilityEngine(
        automaton=auto_bat,
        training_vocabulary=set(train_pat_bat),
        anomaly_threshold=anomaly_threshold,
        window_size=window_size
    )
    explanations_2 = engine_bat_to_skab.explain_sequence(test_pat_skab)
    preds_2 = [1 if e["decision"] == "anomaly" else 0 for e in explanations_2]
    preds_2_padded = [0] * (window_size - 1) + preds_2
    preds_2_upsampled = np.repeat(preds_2_padded, paa_window)
    y_test_skab = df_skab.loc[test_idx_skab, skab_target].values
    if len(preds_2_upsampled) < len(y_test_skab):
        preds_2_upsampled = np.concatenate([preds_2_upsampled, np.zeros(len(y_test_skab) - len(preds_2_upsampled))])
    else:
        preds_2_upsampled = preds_2_upsampled[:len(y_test_skab)]
    m_2 = calculate_metrics(y_test_skab, preds_2_upsampled)

    print(f"Train: SKAB    | Test: BATADAL -> F1: {m_1['f1']:.4f} (Acc: {m_1['accuracy']:.4f}, Prec: {m_1['precision']:.4f}, Rec: {m_1['recall']:.4f})")
    print(f"Train: BATADAL | Test: SKAB    -> F1: {m_2['f1']:.4f} (Acc: {m_2['accuracy']:.4f}, Prec: {m_2['precision']:.4f}, Rec: {m_2['recall']:.4f})")


>>>>>>> c95c420aef4ed2407790f8f0e0e83a6cfcd7ce9f
def run_experiment_pipeline():
    print("=== Zaman Serisi Otomata + DL Deney Sistemi ===")
    pipeline = DataPipeline()
    config = pipeline.config

    batch_size = config["deep_learning"]["batch_size"]
    clf_threshold = config["deep_learning"]["classification_threshold"]
    anomaly_threshold = config["automata"]["anomaly_threshold"]
    paa_window = config["automata"]["paa_window_size"]
    noise_cfg = config["experiment"]["gaussian_noise"]
    skab_val_split = config["experiment"]["skab_train_val_split"]
    unseen_scale = config["experiment"]["unseen_data_scale"]
    window_variants = config["automata"]["window_size_variants"]
    alphabet_variants = config["automata"]["alphabet_size_variants"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_root = os.path.join(project_root, "results")
    plots_dir = os.path.join(results_root, "plots")
    log_path = os.path.join(results_root, "experiment_log.txt")
    logs_csv_path = os.path.join(results_root, "experiment_logs.csv")
    os.makedirs(results_root, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== Deney Logu ===\n")

    experiment_rows = []
    param_analysis_records = []
    datasets = ["BATADAL", "SKAB"]

    for dataset_name in datasets:
        print(f"\n===== VERİ SETİ: {dataset_name} =====")

        if dataset_name == "BATADAL":
            df, feature_cols, target_col = pipeline.load_batadal()
            splits = [pipeline.split_batadal(df)]
        else:
            df, feature_cols, target_col = pipeline.load_skab()
            splits = list(pipeline.get_skab_splits(df))

        for scenario in config["experiment"]["scenarios"]:
            for window_size in window_variants:
                for alphabet_size in alphabet_variants:
                    tag = (
                        f"{dataset_name} | {scenario} | "
                        f"w={window_size} | a={alphabet_size}"
                    )
                    print(f"\n--- {tag} ---")

                    lstm_f1_scores = []
                    cnn_f1_scores = []
                    auto_f1_scores = []
<<<<<<< HEAD
=======
                    auto_unseen_map_accs = []
                    auto_unseen_det_rates = []
>>>>>>> c95c420aef4ed2407790f8f0e0e83a6cfcd7ce9f

                    for seed in config["experiment"]["random_seeds"]:
                        set_seed(seed)
                        seed_lstm_f1, seed_cnn_f1, seed_auto_f1 = [], [], []
                        seed_lstm_all, seed_cnn_all, seed_auto_all = [], [], []

                        for split_idx, split_info in enumerate(splits):
                            if len(split_info) == 3:
                                train_idx, val_idx, test_idx = split_info
                            else:
                                train_val_idx, test_idx = split_info
                                split_point = int(len(train_val_idx) * skab_val_split)
                                train_idx = train_val_idx[:split_point]
                                val_idx = train_val_idx[split_point:]

                            processed = pipeline.preprocess_features(
                                df, train_idx, val_idx, test_idx, feature_cols
                            )
                            dl_data = processed["deep_learning"]
                            auto_data = processed["automata"]

                            X_test_dl = dl_data["test"].copy()
                            X_test_auto = auto_data["test"].copy()
                            if scenario == "gaussian_noise":
                                X_test_dl = add_gaussian_noise(
                                    X_test_dl, noise_cfg["mean"], noise_cfg["std"]
                                )
                                X_test_auto = add_gaussian_noise(
                                    X_test_auto, noise_cfg["mean"], noise_cfg["std"]
                                )
                            elif scenario == "unseen_data":
                                X_test_auto = X_test_auto * unseen_scale

                            y_test = df.loc[test_idx, target_col].values
                            split_label = (
                                f"fold_{split_idx}" if dataset_name == "SKAB" else "full"
                            )
                            results_dir = os.path.join(
                                results_root,
                                dataset_name,
                                scenario,
                                f"w{window_size}_a{alphabet_size}",
                                f"seed_{seed}",
                                split_label,
                            )
                            os.makedirs(results_dir, exist_ok=True)

                            train_ds = TimeSeriesDataset(
                                dl_data["train"],
                                df.loc[train_idx, target_col].values,
                                window_size,
                            )
                            val_ds = TimeSeriesDataset(
                                dl_data["val"],
                                df.loc[val_idx, target_col].values,
                                window_size,
                            )
                            test_ds = TimeSeriesDataset(X_test_dl, y_test, window_size)
                            train_loader = DataLoader(
                                train_ds, batch_size=batch_size, shuffle=False
                            )
                            val_loader = DataLoader(
                                val_ds, batch_size=batch_size, shuffle=False
                            )
                            test_loader = DataLoader(
                                test_ds, batch_size=batch_size, shuffle=False
                            )

<<<<<<< HEAD
=======
                            y_val = df.loc[val_idx, target_col].values

                            # === LSTM MODEL ===
>>>>>>> c95c420aef4ed2407790f8f0e0e83a6cfcd7ce9f
                            lstm = LSTMAnomalyDetector(
                                input_size=dl_data["train"].shape[1],
                                hidden_size=config["deep_learning"]["lstm"]["hidden_size"],
                                num_layers=config["deep_learning"]["lstm"]["num_layers"],
                                dropout=config["deep_learning"]["lstm"]["dropout"],
                            )
<<<<<<< HEAD
                            lstm = train_model(lstm, train_loader, val_loader, config, seed)
                            lstm_probs, lstm_binary = evaluate_model_predictions(
                                lstm, test_loader, window_size, device, clf_threshold
                            )

=======
                            start_time = time.time()
                            lstm = train_model(lstm, train_loader, val_loader, config, seed)
                            lstm_train_time = time.time() - start_time

                            # Optimize threshold on validation set
                            lstm_val_probs = evaluate_val_predictions(lstm, val_loader, window_size, device)
                            lstm_opt_thresh = optimize_threshold(lstm_val_probs, y_val)
                            if lstm_opt_thresh is None or lstm_opt_thresh <= 0:
                                lstm_opt_thresh = clf_threshold

                            start_time = time.time()
                            lstm_probs, lstm_binary = evaluate_model_predictions(
                                lstm, test_loader, window_size, device, lstm_opt_thresh
                            )
                            lstm_infer_time = time.time() - start_time

                            # === CNN MODEL ===
>>>>>>> c95c420aef4ed2407790f8f0e0e83a6cfcd7ce9f
                            cnn = CNN1DAnomalyDetector(
                                input_size=dl_data["train"].shape[1],
                                filters=config["deep_learning"]["cnn_1d"]["filters"],
                                kernel_size=config["deep_learning"]["cnn_1d"]["kernel_size"],
                            )
<<<<<<< HEAD
                            cnn = train_model(cnn, train_loader, val_loader, config, seed)
                            cnn_probs, cnn_binary = evaluate_model_predictions(
                                cnn, test_loader, window_size, device, clf_threshold
                            )

=======
                            start_time = time.time()
                            cnn = train_model(cnn, train_loader, val_loader, config, seed)
                            cnn_train_time = time.time() - start_time

                            # Optimize threshold on validation set
                            cnn_val_probs = evaluate_val_predictions(cnn, val_loader, window_size, device)
                            cnn_opt_thresh = optimize_threshold(cnn_val_probs, y_val)
                            if cnn_opt_thresh is None or cnn_opt_thresh <= 0:
                                cnn_opt_thresh = clf_threshold

                            start_time = time.time()
                            cnn_probs, cnn_binary = evaluate_model_predictions(
                                cnn, test_loader, window_size, device, cnn_opt_thresh
                            )
                            cnn_infer_time = time.time() - start_time

                            # === AUTOMATA MODEL ===
                            start_time = time.time()
>>>>>>> c95c420aef4ed2407790f8f0e0e83a6cfcd7ce9f
                            train_patterns = pc1_to_patterns(
                                auto_data["train"],
                                paa_window,
                                alphabet_size,
                                window_size,
                            )
<<<<<<< HEAD
                            automaton = build_transition_matrix([train_patterns])
=======
                            smoothing_k = config["automata"].get("smoothing_k", 0.1)
                            automaton = build_transition_matrix([train_patterns], smoothing_k=smoothing_k)
                            auto_train_time = time.time() - start_time

>>>>>>> c95c420aef4ed2407790f8f0e0e83a6cfcd7ce9f
                            state_count, transition_density = compute_automata_stats(
                                automaton
                            )
                            param_analysis_records.append(
                                {
                                    "dataset": dataset_name,
                                    "scenario": scenario,
                                    "window_size": window_size,
                                    "alphabet_size": alphabet_size,
                                    "paa_window": paa_window,
                                    "seed": seed,
                                    "split": split_label,
                                    "state_count": state_count,
                                    "transition_density": transition_density,
                                }
                            )

<<<<<<< HEAD
=======
                            start_time = time.time()
>>>>>>> c95c420aef4ed2407790f8f0e0e83a6cfcd7ce9f
                            test_patterns = pc1_to_patterns(
                                X_test_auto, paa_window, alphabet_size, window_size
                            )
                            engine = ExplainabilityEngine(
                                automaton=automaton,
                                training_vocabulary=set(train_patterns),
                                anomaly_threshold=anomaly_threshold,
<<<<<<< HEAD
=======
                                window_size=window_size,
>>>>>>> c95c420aef4ed2407790f8f0e0e83a6cfcd7ce9f
                            )
                            engine.save_json(
                                test_patterns,
                                os.path.join(results_dir, "automata_explanation.json"),
                            )
                            explanations = engine.explain_sequence(test_patterns)
<<<<<<< HEAD
                            auto_preds = [
                                1 if e["decision"] == "anomaly" else 0
                                for e in explanations
                            ]
                            auto_preds = [0] * (window_size - 1) + auto_preds

                            lstm_m = calculate_metrics(
                                y_test, lstm_binary, threshold=clf_threshold
                            )
                            cnn_m = calculate_metrics(
                                y_test, cnn_binary, threshold=clf_threshold
                            )
                            auto_m = calculate_metrics(y_test, auto_preds)

                            for model_name, metrics in [
                                ("LSTM", lstm_m),
                                ("CNN", cnn_m),
                                ("Automata", auto_m),
=======
                            auto_preds_sax = [
                                1 if e["decision"] == "anomaly" else 0
                                for e in explanations
                            ]
                            
                            # Align/upsample Automata predictions to original time series resolution
                            auto_preds_sax_padded = [0] * (window_size - 1) + auto_preds_sax
                            auto_preds_upsampled = np.repeat(auto_preds_sax_padded, paa_window)
                            
                            if len(auto_preds_upsampled) < len(y_test):
                                diff = len(y_test) - len(auto_preds_upsampled)
                                auto_preds_upsampled = np.concatenate([auto_preds_upsampled, np.zeros(diff)])
                            elif len(auto_preds_upsampled) > len(y_test):
                                auto_preds_upsampled = auto_preds_upsampled[:len(y_test)]
                                
                            auto_preds = list(auto_preds_upsampled.astype(int))
                            auto_infer_time = time.time() - start_time

                            # Unseen scenario metrics: Mapping Accuracy & Detection Rate
                            mapping_accuracy = "N/A"
                            detection_rate = "N/A"
                            if scenario == "unseen_data":
                                unseen_mask = np.zeros(len(y_test), dtype=bool)
                                for t_idx in range(len(y_test)):
                                    k_idx = t_idx // paa_window
                                    p_idx = k_idx - window_size + 1
                                    if 0 <= p_idx < len(explanations):
                                        if explanations[p_idx]["status"] == "unseen":
                                            unseen_mask[t_idx] = True
                                if np.sum(unseen_mask) > 0:
                                    unseen_y_true = y_test[unseen_mask]
                                    unseen_preds = np.array(auto_preds)[unseen_mask]
                                    mapping_accuracy = float(np.mean(unseen_preds == unseen_y_true))
                                    tp = np.sum((unseen_preds == 1) & (unseen_y_true == 1))
                                    fn = np.sum((unseen_preds == 0) & (unseen_y_true == 1))
                                    detection_rate = float(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
                                    auto_unseen_map_accs.append(mapping_accuracy)
                                    auto_unseen_det_rates.append(detection_rate)

                            # === METRIC CALCULATION ===
                            lstm_m = calculate_metrics(
                                y_test, lstm_binary, threshold=lstm_opt_thresh
                            )
                            cnn_m = calculate_metrics(
                                y_test, cnn_binary, threshold=cnn_opt_thresh
                            )
                            auto_m = calculate_metrics(y_test, auto_preds)

                            for model_name, metrics, t_time, i_time, map_acc, det_rt in [
                                ("LSTM", lstm_m, lstm_train_time, lstm_infer_time, "N/A", "N/A"),
                                ("CNN", cnn_m, cnn_train_time, cnn_infer_time, "N/A", "N/A"),
                                ("Automata", auto_m, auto_train_time, auto_infer_time, mapping_accuracy, detection_rate),
>>>>>>> c95c420aef4ed2407790f8f0e0e83a6cfcd7ce9f
                            ]:
                                _append_metric_row(
                                    experiment_rows,
                                    dataset_name,
                                    scenario,
                                    window_size,
                                    alphabet_size,
                                    seed,
                                    split_label,
                                    model_name,
                                    metrics,
                                    state_count,
                                    transition_density,
<<<<<<< HEAD
=======
                                    training_time=t_time,
                                    inference_time=i_time,
                                    mapping_accuracy=map_acc,
                                    detection_rate=det_rt,
>>>>>>> c95c420aef4ed2407790f8f0e0e83a6cfcd7ce9f
                                )

                            seed_lstm_f1.append(lstm_m["f1"])
                            seed_cnn_f1.append(cnn_m["f1"])
                            seed_auto_f1.append(auto_m["f1"])
                            seed_lstm_all.append(lstm_m)
                            seed_cnn_all.append(cnn_m)
                            seed_auto_all.append(auto_m)

                            lbl = f"{dataset_name}-{scenario}-w{window_size}-s{seed}"
<<<<<<< HEAD
                            plot_confusion_matrix(
                                y_test, lstm_binary, f"LSTM-{lbl}", results_dir
                            )
                            plot_confusion_matrix(
                                y_test, cnn_binary, f"CNN1D-{lbl}", results_dir
                            )
                            plot_confusion_matrix(
                                y_test, auto_preds, f"Automata-{lbl}", results_dir
                            )
=======
                            plot_confusion_matrix(y_test, lstm_binary, f"LSTM-{lbl}", results_dir)
                            plot_confusion_matrix(y_test, cnn_binary, f"CNN1D-{lbl}", results_dir)
                            plot_confusion_matrix(y_test, auto_preds, f"Automata-{lbl}", results_dir)

>>>>>>> c95c420aef4ed2407790f8f0e0e83a6cfcd7ce9f
                            plot_roc_curve(y_test, lstm_probs, f"LSTM-{lbl}", results_dir)
                            plot_roc_curve(y_test, cnn_probs, f"CNN1D-{lbl}", results_dir)
                            plot_automata_state_diagram(
                                automaton.to_dataframe(), save_dir=results_dir
                            )
                            plot_transition_heatmap(
                                automaton.to_dataframe(), save_dir=results_dir
                            )

                        if dataset_name == "SKAB":
                            _log_line(
                                log_path,
                                f"[SKAB fold ort.] {tag} seed={seed} | "
                                f"LSTM F1: {np.mean(seed_lstm_f1):.4f} ± {np.std(seed_lstm_f1):.4f} | "
                                f"CNN F1: {np.mean(seed_cnn_f1):.4f} ± {np.std(seed_cnn_f1):.4f} | "
                                f"Otomata F1: {np.mean(seed_auto_f1):.4f} ± {np.std(seed_auto_f1):.4f}",
                            )

                        lstm_f1_scores.append(float(np.mean(seed_lstm_f1)))
                        cnn_f1_scores.append(float(np.mean(seed_cnn_f1)))
                        auto_f1_scores.append(float(np.mean(seed_auto_f1)))

<<<<<<< HEAD
                    summary = (
                        f">>> RAPOR ({tag}) <<<\n"
                        f"LSTM  : {np.mean(lstm_f1_scores):.4f} ± {np.std(lstm_f1_scores):.4f}\n"
                        f"CNN   : {np.mean(cnn_f1_scores):.4f} ± {np.std(cnn_f1_scores):.4f}\n"
                        f"Otomata: {np.mean(auto_f1_scores):.4f} ± {np.std(auto_f1_scores):.4f}"
=======
                    # Calculate mean training/inference times for summary log
                    lstm_train_times = [
                        r["training_time"] for r in experiment_rows 
                        if r["model"] == "LSTM" and r["dataset"] == dataset_name 
                        and r["window_size"] == window_size and r["alphabet_size"] == alphabet_size
                    ]
                    lstm_infer_times = [
                        r["inference_time"] for r in experiment_rows 
                        if r["model"] == "LSTM" and r["dataset"] == dataset_name 
                        and r["window_size"] == window_size and r["alphabet_size"] == alphabet_size
                    ]
                    cnn_train_times = [
                        r["training_time"] for r in experiment_rows 
                        if r["model"] == "CNN" and r["dataset"] == dataset_name 
                        and r["window_size"] == window_size and r["alphabet_size"] == alphabet_size
                    ]
                    cnn_infer_times = [
                        r["inference_time"] for r in experiment_rows 
                        if r["model"] == "CNN" and r["dataset"] == dataset_name 
                        and r["window_size"] == window_size and r["alphabet_size"] == alphabet_size
                    ]
                    auto_train_times = [
                        r["training_time"] for r in experiment_rows 
                        if r["model"] == "Automata" and r["dataset"] == dataset_name 
                        and r["window_size"] == window_size and r["alphabet_size"] == alphabet_size
                    ]
                    auto_infer_times = [
                        r["inference_time"] for r in experiment_rows 
                        if r["model"] == "Automata" and r["dataset"] == dataset_name 
                        and r["window_size"] == window_size and r["alphabet_size"] == alphabet_size
                    ]

                    unseen_metrics_str = ""
                    if scenario == "unseen_data" and auto_unseen_map_accs:
                        unseen_metrics_str = (
                            f" | Otomata Unseen Map. Acc: {np.mean(auto_unseen_map_accs):.4f} "
                            f"| Det. Rate: {np.mean(auto_unseen_det_rates):.4f}"
                        )

                    summary = (
                        f">>> RAPOR ({tag}) <<<\n"
                        f"LSTM   - F1: {np.mean(lstm_f1_scores):.4f} ± {np.std(lstm_f1_scores):.4f} | Egitim: {np.mean(lstm_train_times):.2f}s | Cikarim: {np.mean(lstm_infer_times):.4f}s\n"
                        f"CNN    - F1: {np.mean(cnn_f1_scores):.4f} ± {np.std(cnn_f1_scores):.4f} | Egitim: {np.mean(cnn_train_times):.2f}s | Cikarim: {np.mean(cnn_infer_times):.4f}s\n"
                        f"Otomata - F1: {np.mean(auto_f1_scores):.4f} ± {np.std(auto_f1_scores):.4f} | Egitim: {np.mean(auto_train_times):.4f}s | Cikarim: {np.mean(auto_infer_times):.4f}s{unseen_metrics_str}"
>>>>>>> c95c420aef4ed2407790f8f0e0e83a6cfcd7ce9f
                    )
                    _log_line(log_path, summary)

                    w_lstm = run_wilcoxon_paired_f1(lstm_f1_scores, auto_f1_scores)
                    w_cnn = run_wilcoxon_paired_f1(cnn_f1_scores, auto_f1_scores)
                    _log_line(
                        log_path,
                        f"[Wilcoxon F1] {tag}\n"
                        f"  LSTM vs Otomata p={w_lstm['p_value']:.6e} "
                        f"significant={w_lstm['significant']}\n"
                        f"  CNN vs Otomata  p={w_cnn['p_value']:.6e} "
                        f"significant={w_cnn['significant']}",
                    )

<<<<<<< HEAD
=======
    run_cross_dataset_automata(pipeline, config)

>>>>>>> c95c420aef4ed2407790f8f0e0e83a6cfcd7ce9f
    _save_experiment_logs(experiment_rows, logs_csv_path)
    with open(os.path.join(results_root, "experiment_logs.json"), "w", encoding="utf-8") as f:
        json.dump(experiment_rows, f, indent=2, ensure_ascii=False)

    analysis_path = os.path.join(results_root, "param_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(param_analysis_records, f, indent=2, ensure_ascii=False)

    plot_parameter_sensitivity(experiment_rows, param_analysis_records, plots_dir)
    print(f"\nLoglar: {logs_csv_path}")
    print(f"Grafikler: {plots_dir}")


if __name__ == "__main__":
    run_experiment_pipeline()
