import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import networkx as nx
import pandas as pd

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir="results"):
    """Confusion Matrix (Karmaşıklık Matrisi) çizer."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5), facecolor='#F8F9FA')
    ax = plt.gca()
    ax.set_facecolor('#F8F9FA')
    sns.heatmap(cm, annot=True, fmt="d", cmap="mako", cbar=False,
                linewidths=1, linecolor='#F8F9FA',
                annot_kws={"size": 12, "weight": "bold"})
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, pad=15, color='#2C3E50', weight='bold')
    plt.xlabel("Tahmin Edilen (Predicted)", fontsize=11, color='#34495E', labelpad=10)
    plt.ylabel("Gerçek (Actual)", fontsize=11, color='#34495E', labelpad=10)
    plt.xticks(color='#7F8C8D')
    plt.yticks(color='#7F8C8D')
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{model_name}_confusion_matrix.png"), dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()

def plot_roc_curve(y_true, y_probs, model_name, save_dir="results"):
    """ROC eğrisi çizer ve AUC (Area Under Curve) değerini hesaplar."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5), facecolor='#F8F9FA')
    ax = plt.gca()
    ax.set_facecolor('#FFFFFF')
    ax.grid(color='#E5E7EB', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.plot(fpr, tpr, color='#E74C3C', lw=2.5, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='#34495E', lw=2, linestyle=':')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11, color='#34495E', weight='500')
    plt.ylabel('True Positive Rate', fontsize=11, color='#34495E', weight='500')
    plt.title(f'ROC Eğrisi - {model_name}', fontsize=14, pad=15, color='#2C3E50', weight='bold')
    plt.legend(loc="lower right", frameon=True, facecolor='#FFFFFF', edgecolor='#BDC3C7')
    
    for spine in ax.spines.values():
        spine.set_color('#BDC3C7')
        
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{model_name}_roc_curve.png"), dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()

def plot_automata_state_diagram(transition_matrix_df: pd.DataFrame, save_dir="results"):
    """Otomata modelinin durum geçişlerini NetworkX kullanarak görselleştirir."""
    G = nx.DiGraph()
    for src_state in transition_matrix_df.index:
        for dst_state in transition_matrix_df.columns:
            prob = transition_matrix_df.loc[src_state, dst_state]
            if prob > 0.0:
                G.add_edge(src_state, dst_state, weight=prob)
                
    plt.figure(figsize=(10, 8), facecolor='#F8F9FA')
    pos = nx.spring_layout(G, k=0.9, iterations=80, seed=42)
    
    edges = G.edges()
    weights = [G[u][v]['weight'] * 6 for u, v in edges]
    
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color='#3498DB', 
                           edgecolors='#2980B9', linewidths=2, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight="bold", font_color='white')
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, 
                           arrowstyle="-|>", arrowsize=22, edge_color="#95A5A6", connectionstyle='arc3,rad=0.15')
    
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, 
                                 bbox=dict(facecolor='#FFFFFF', edgecolor='none', alpha=0.7))
    
    plt.title("Probabilistic Automata - State Transitions", fontsize=16, pad=20, color='#2C3E50', weight='bold')
    plt.axis("off")
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "automata_state_diagram.png"), dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()

def plot_transition_heatmap(transition_matrix_df: pd.DataFrame, save_dir="results"):
    """Otomata geçiş olasılıklarının Isı Haritasını (Heatmap) çizer."""
    plt.figure(figsize=(8, 6), facecolor='#F8F9FA')
    ax = plt.gca()
    ax.set_facecolor('#F8F9FA')
    sns.heatmap(transition_matrix_df, annot=True, fmt=".2f", cmap="crest", 
                linewidths=0.5, linecolor='#FFFFFF',
                cbar_kws={'label': 'Geçiş Olasılığı'})
    plt.title("Otomata Geçiş Olasılıkları Isı Haritası", fontsize=14, pad=15, color='#2C3E50', weight='bold')
    plt.xlabel("Hedef Durum (To State)", fontsize=11, color='#34495E', labelpad=10)
    plt.ylabel("Kaynak Durum (From State)", fontsize=11, color='#34495E', labelpad=10)
    plt.xticks(rotation=45, ha='right', color='#7F8C8D')
    plt.yticks(rotation=0, color='#7F8C8D')
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "automata_transition_heatmap.png"), dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
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
