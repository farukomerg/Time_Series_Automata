import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import networkx as nx
import pandas as pd

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir="results"):
    """Confusion Matrix (Karmaşıklık Matrisi) çizer."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Tahmin Edilen (Predicted)")
    plt.ylabel("Gerçek (Actual)")
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()

def plot_roc_curve(y_true, y_probs, model_name, save_dir="results"):
    """ROC eğrisi çizer ve AUC (Area Under Curve) değerini hesaplar."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Area = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Eğrisi - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{model_name}_roc_curve.png"))
    plt.close()

def plot_automata_state_diagram(transition_matrix_df: pd.DataFrame, save_dir="results"):
    """
    Otomata modelinin durum geçişlerini NetworkX kullanarak görselleştirir.
    Rubric'te beklenen Durum Geçiş (State Transition) grafiğini PNG olarak çıkarır.
    """
    G = nx.DiGraph()
    
    # DataFrame üzerinden matris değerleriyle node ve edge oluştur
    for src_state in transition_matrix_df.index:
        for dst_state in transition_matrix_df.columns:
            prob = transition_matrix_df.loc[src_state, dst_state]
            if prob > 0.0:  # Sadece sıfırdan büyük geçişleri çiz
                G.add_edge(src_state, dst_state, weight=prob)
                
    plt.figure(figsize=(10, 8))
    # Düğümleri yaylı (spring) yerleşim planıyla dağıt
    pos = nx.spring_layout(G, k=0.8, iterations=50)
    
    # Kenar kalınlıklarını olasılıklara bağla
    edges = G.edges()
    weights = [G[u][v]['weight'] * 4 for u, v in edges]
    
    nx.draw_networkx_nodes(G, pos, node_size=1800, node_color='lightblue', edgecolors='gray')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, arrowstyle="-|>", arrowsize=20, edge_color="gray")
    
    # Olasılık değerlerini okların üzerine yaz
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Probabilistic Automata - State Transition Diagram", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "automata_state_diagram.png"))
    plt.close()

def plot_transition_heatmap(transition_matrix_df: pd.DataFrame, save_dir="results"):
    """Otomata geçiş olasılıklarının Isı Haritasını (Heatmap) çizer."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(transition_matrix_df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Otomata Geçiş Olasılıkları Isı Haritası")
    plt.xlabel("Hedef Durum (To State)")
    plt.ylabel("Kaynak Durum (From State)")
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "automata_transition_heatmap.png"))
    plt.close()


def plot_parameter_sensitivity(experiment_logs, param_records, save_dir="results/plots"):
    """Window/Alphabet size etkisini F1 ve state sayısı üzerinde çizer."""
    os.makedirs(save_dir, exist_ok=True)
    if not experiment_logs:
        return

    df = pd.DataFrame(experiment_logs)
    pdf = pd.DataFrame(param_records) if param_records else pd.DataFrame()

    for dataset in df["dataset"].unique():
        ddf = df[df["dataset"] == dataset]

        for scenario in ddf["scenario"].unique():
            sdf = ddf[ddf["scenario"] == scenario]
            tag = f"{dataset}_{scenario}"

            fig, ax = plt.subplots(figsize=(8, 5))
            for model in sdf["model"].unique():
                mdf = sdf[sdf["model"] == model]
                agg = mdf.groupby("window_size")["f1"].mean().reset_index()
                ax.plot(agg["window_size"], agg["f1"], marker="o", label=model)
            ax.set_xlabel("Window Size")
            ax.set_ylabel("F1 Score")
            ax.set_title(f"Parametre Duyarliligi - Window Size ({tag})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{tag}_f1_vs_window_size.png"))
            plt.close()

            fig, ax = plt.subplots(figsize=(8, 5))
            for model in sdf["model"].unique():
                mdf = sdf[sdf["model"] == model]
                agg = mdf.groupby("alphabet_size")["f1"].mean().reset_index()
                ax.plot(agg["alphabet_size"], agg["f1"], marker="s", label=model)
            ax.set_xlabel("Alphabet Size")
            ax.set_ylabel("F1 Score")
            ax.set_title(f"Parametre Duyarliligi - Alphabet Size ({tag})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{tag}_f1_vs_alphabet_size.png"))
            plt.close()

            if not pdf.empty:
                psub = pdf[(pdf["dataset"] == dataset) & (pdf["scenario"] == scenario)]
                if not psub.empty:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    ws = psub.groupby("window_size")["state_count"].mean()
                    axes[0].bar(ws.index.astype(str), ws.values, color="steelblue")
                    axes[0].set_title("State Sayisi vs Window Size")
                    axes[0].set_xlabel("Window Size")
                    axes[0].set_ylabel("Ortalama State Sayisi")

                    als = psub.groupby("alphabet_size")["state_count"].mean()
                    axes[1].bar(als.index.astype(str), als.values, color="coral")
                    axes[1].set_title("State Sayisi vs Alphabet Size")
                    axes[1].set_xlabel("Alphabet Size")
                    axes[1].set_ylabel("Ortalama State Sayisi")
                    plt.suptitle(f"Otomata State Analizi ({tag})")
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f"{tag}_state_count_sensitivity.png"))
                    plt.close()
