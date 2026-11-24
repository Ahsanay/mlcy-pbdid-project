"""
Step 4: figures and result tables from the saved JSON files.

Run:  python src/make_figures.py
Writes PNG figures and results/summary_tables.md
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prepare_data import ROOT

RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")


def load(name):
    with open(os.path.join(RES, name, "results.json")) as f:
        return json.load(f)


def confusion(ax, cm, names, title):
    cm = np.array(cm, dtype=float)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{pct[i, j]:.1f}", ha="center", va="center",
                    color="white" if pct[i, j] > 50 else "black", fontsize=9)


def main():
    os.makedirs(FIG, exist_ok=True)
    full, noid, ext = load("reproduction_full"), load("reproduction_noid"), load("extension")
    lines = ["# Result tables", ""]

    # ---- reproduction
    paper = {"Normal_vs_DDoS": "above 99", "Normal_vs_DoS": "above 99",
             "DDoS_vs_DoS": "above 97", "Three_class": "96.3"}
    lines += ["## Reproduction: accuracy (%), mean and standard deviation over 4 chunks", "",
              "| Task | Paper | Ours, 26 features | Ours, 22 features (no time, no IP) |",
              "|---|---|---|---|"]
    for t in full["tasks"]:
        a, b = full["tasks"][t], noid["tasks"][t]
        lines.append(f"| {t.replace('_', ' ')} | {paper[t]} | "
                     f"{a['mean_accuracy'] * 100:.2f} ± {a['std_accuracy'] * 100:.2f} | "
                     f"{b['mean_accuracy'] * 100:.2f} ± {b['std_accuracy'] * 100:.2f} |")
    lines += ["", "Three-class, per-class precision / recall / F1 (26 features):", "",
              "| Class | Precision | Recall | F1 |", "|---|---|---|---|"]
    t = full["tasks"]["Three_class"]
    for i, c in enumerate(t["classes"]):
        lines.append(f"| {c} | {t['mean_precision'][i]:.3f} | "
                     f"{t['mean_recall'][i]:.3f} | {t['mean_f1'][i]:.3f} |")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    confusion(axes[0], full["tasks"]["Three_class"]["confusion_matrix_sum"],
              full["tasks"]["Three_class"]["classes"], "All 26 features (% of true class)")
    confusion(axes[1], noid["tasks"]["Three_class"]["confusion_matrix_sum"],
              noid["tasks"]["Three_class"]["classes"], "Without time and IP features")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_reproduction_confusion.png"), dpi=200)
    plt.close(fig)

    # ---- A: imbalance handling
    a = ext["A_imbalance"]
    label = {"plain": "No handling", "undersample": "Equal count\n(paper)",
             "weighted": "Class\nweights", "smote": "SMOTE", "focal": "Focal\nloss"}
    methods = list(a["methods"])
    lines += ["", "## Extension A: five-class Bot-IoT, mean over 5 seeds", "",
              "Class counts used: " + ", ".join(f"{k} {v:,}" for k, v in a["class_counts"].items()),
              "", "| Method | Accuracy | Macro F1 | Macro PR-AUC | Normal recall | Normal precision | Theft recall | Theft precision |",
              "|---|---|---|---|---|---|---|---|"]
    for m in methods:
        v = a["methods"][m]
        lines.append(f"| {label[m].replace(chr(10), ' ')} | {v['accuracy']['mean']:.4f} | "
                     f"{v['macro_f1']['mean']:.3f} ± {v['macro_f1']['std']:.3f} | "
                     f"{v['macro_pr_auc']['mean']:.3f} | "
                     f"{v['recall']['Normal']['mean']:.3f} | {v['precision']['Normal']['mean']:.3f} | "
                     f"{v['recall']['Theft']['mean']:.3f} | {v['precision']['Theft']['mean']:.3f} |")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    x = np.arange(len(methods))
    axes[0].bar(x - 0.2, [a["methods"][m]["accuracy"]["mean"] for m in methods], 0.4, label="Accuracy")
    axes[0].bar(x + 0.2, [a["methods"][m]["macro_f1"]["mean"] for m in methods], 0.4,
                yerr=[a["methods"][m]["macro_f1"]["std"] for m in methods], label="Macro F1")
    axes[0].set_xticks(x, [label[m] for m in methods], fontsize=8)
    axes[0].set_ylim(0.5, 1.02)
    axes[0].set_title("Accuracy and macro F1", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    w = 0.2
    for k, (cls, kind) in enumerate([("Normal", "recall"), ("Normal", "precision"),
                                     ("Theft", "recall"), ("Theft", "precision")]):
        axes[1].bar(x + (k - 1.5) * w, [a["methods"][m][kind][cls]["mean"] for m in methods],
                    w, label=f"{cls} {kind}")
    axes[1].set_xticks(x, [label[m] for m in methods], fontsize=8)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Minority classes", fontsize=10)
    axes[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_imbalance_methods.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    confusion(axes[0], a["methods"]["plain"]["confusion_matrix_sum"], a["classes"], "No imbalance handling")
    confusion(axes[1], a["methods"]["smote"]["confusion_matrix_sum"], a["classes"], "SMOTE on the training split")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_five_class_confusion.png"), dpi=200)
    plt.close(fig)

    # ---- B: cross-dataset
    b = ext["B_cross_dataset"]
    lines += ["", "## Extension B: cross-dataset (Normal, DoS, Reconnaissance), mean over 5 seeds", "",
              "| Setting | Accuracy | Macro F1 | Normal recall | DoS recall | Reconnaissance recall |",
              "|---|---|---|---|---|---|"]
    for k, v in b["runs"].items():
        lines.append(f"| {k} | {v['accuracy']['mean']:.3f} | {v['macro_f1']['mean']:.3f} | "
                     + " | ".join(f"{v['recall'][c]['mean']:.3f}" for c in b["classes"]) + " |")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    keys = list(b["runs"])
    ax.bar(range(len(keys)), [b["runs"][k]["macro_f1"]["mean"] for k in keys],
           color=["tab:blue", "tab:red", "tab:blue", "tab:red"])
    ax.set_xticks(range(len(keys)), [k.replace(", ", "\n") for k in keys], fontsize=8)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("In-dataset (blue) against cross-dataset (red)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_cross_dataset.png"), dpi=200)
    plt.close(fig)

    # ---- C: cost
    lines += ["", "## Extension C: model cost (CPU, one thread)", "",
              "| Model | Parameters | Microseconds per record |", "|---|---|---|"]
    for k, v in ext["C_cost"].items():
        lines.append(f"| {k} | {v['parameters']:,} | {v['ms_per_record_cpu_1_thread'] * 1000:.2f} |")
    lq = ext["C_light_model_quality"]
    lines += ["", f"Light network with class weights, five-class Bot-IoT: accuracy "
              f"{lq['accuracy']['mean']:.4f}, macro F1 {lq['macro_f1']['mean']:.3f}."]

    with open(os.path.join(RES, "summary_tables.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
