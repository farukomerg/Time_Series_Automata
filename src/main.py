import os
import json
import csv
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
from metrics import calculate_metrics, run_wilcoxon_test, run_wilcoxon_paired_f1
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

                            lstm = LSTMAnomalyDetector(
                                input_size=dl_data["train"].shape[1],
                                hidden_size=config["deep_learning"]["lstm"]["hidden_size"],
                                num_layers=config["deep_learning"]["lstm"]["num_layers"],
                                dropout=config["deep_learning"]["lstm"]["dropout"],
                            )
                            lstm = train_model(lstm, train_loader, val_loader, config, seed)
                            lstm_probs, lstm_binary = evaluate_model_predictions(
                                lstm, test_loader, window_size, device, clf_threshold
                            )

                            cnn = CNN1DAnomalyDetector(
                                input_size=dl_data["train"].shape[1],
                                filters=config["deep_learning"]["cnn_1d"]["filters"],
                                kernel_size=config["deep_learning"]["cnn_1d"]["kernel_size"],
                            )
                            cnn = train_model(cnn, train_loader, val_loader, config, seed)
                            cnn_probs, cnn_binary = evaluate_model_predictions(
                                cnn, test_loader, window_size, device, clf_threshold
                            )

                            train_patterns = pc1_to_patterns(
                                auto_data["train"],
                                paa_window,
                                alphabet_size,
                                window_size,
                            )
                            automaton = build_transition_matrix([train_patterns])
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

                            test_patterns = pc1_to_patterns(
                                X_test_auto, paa_window, alphabet_size, window_size
                            )
                            engine = ExplainabilityEngine(
                                automaton=automaton,
                                training_vocabulary=set(train_patterns),
                                anomaly_threshold=anomaly_threshold,
                            )
                            engine.save_json(
                                test_patterns,
                                os.path.join(results_dir, "automata_explanation.json"),
                            )
                            explanations = engine.explain_sequence(test_patterns)
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
                                )

                            seed_lstm_f1.append(lstm_m["f1"])
                            seed_cnn_f1.append(cnn_m["f1"])
                            seed_auto_f1.append(auto_m["f1"])
                            seed_lstm_all.append(lstm_m)
                            seed_cnn_all.append(cnn_m)
                            seed_auto_all.append(auto_m)

                            lbl = f"{dataset_name}-{scenario}-w{window_size}-s{seed}"
                            plot_confusion_matrix(
                                y_test, lstm_binary, f"LSTM-{lbl}", results_dir
                            )
                            plot_confusion_matrix(
                                y_test, cnn_binary, f"CNN1D-{lbl}", results_dir
                            )
                            plot_confusion_matrix(
                                y_test, auto_preds, f"Automata-{lbl}", results_dir
                            )
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

                    summary = (
                        f">>> RAPOR ({tag}) <<<\n"
                        f"LSTM  : {np.mean(lstm_f1_scores):.4f} ± {np.std(lstm_f1_scores):.4f}\n"
                        f"CNN   : {np.mean(cnn_f1_scores):.4f} ± {np.std(cnn_f1_scores):.4f}\n"
                        f"Otomata: {np.mean(auto_f1_scores):.4f} ± {np.std(auto_f1_scores):.4f}"
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
