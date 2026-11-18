"""Regenerate manuscript performance figures from the matched six-window outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.sankey import Sankey
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
BASE = ROOT / "runs" / "pamap2_main"
# The manuscript TCST-HAR checkpoint is the 128-dimensional, 548,490-parameter model.
TCST = ROOT / "runs" / "pamap2_main" / "tcst_har"
MODELS = ["tcst_har", "grufcn", "cnn1d", "resnet1d", "timesnet", "moderntcn"]
LABELS = ["Lying", "Sitting", "Standing", "Walking", "Running", "Cycling", "Nordic walk", "Asc. stairs", "Desc. stairs", "Rope jump"]
COLORS = {"tcst_har": "#B2182B", "grufcn": "#2166AC", "cnn1d": "#4D9221", "resnet1d": "#984EA3", "timesnet": "#E66101", "moderntcn": "#1B9E77"}


def draw_confusion(ax, cm, title, show_colorbar=False):
    """Draw a readable recall-normalized matrix without a wall of zero labels."""
    im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
    ax.set_title(title, fontsize=9, weight="bold", pad=7)
    ax.set_xticks(np.arange(len(LABELS)), LABELS, rotation=48, ha="right", fontsize=6.2)
    ax.set_yticks(np.arange(len(LABELS)), LABELS, fontsize=6.2)
    ax.tick_params(length=0, pad=2)
    ax.set_xlabel("Predicted", fontsize=7, labelpad=3)
    ax.set_ylabel("True", fontsize=7, labelpad=3)
    ax.set_aspect("equal")
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            value = float(cm[row, col])
            # Zero cells are intentionally blank; non-zero errors remain visible.
            if value <= 1e-12:
                continue
            if row == col and value >= 0.9995:
                text = "100"
            else:
                text = f"{value * 100:.1f}"
            color = "white" if value >= 0.55 else "#12355B"
            ax.text(col, row, text, ha="center", va="center", fontsize=6.1, color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if show_colorbar:
        return im
    return None


def prediction(model):
    folder = TCST if model == "tcst_har" else BASE / model
    raw = np.load(folder / "predictions.npz", allow_pickle=False)
    return raw["y_true"], raw["y_pred"], raw["y_prob"]


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    truth, _, _ = prediction("tcst_har")
    classes = np.arange(len(LABELS))
    ybin = label_binarize(truth, classes=classes)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)
    for model in MODELS:
        y, _, prob = prediction(model)
        fpr, tpr, _ = roc_curve(ybin.ravel(), prob.ravel())
        precision, recall, _ = precision_recall_curve(ybin.ravel(), prob.ravel())
        label = model.replace("_", " ").upper()
        axes[0].plot(fpr, tpr, label=label, color=COLORS[model], lw=1.2)
        axes[1].plot(recall, precision, label=label, color=COLORS[model], lw=1.2)
    axes[0].set_title("Micro-averaged ROC")
    axes[1].set_title("Micro-averaged precision-recall")
    for ax in axes:
        ax.legend(fontsize=6, frameon=False, loc="lower right")
        ax.grid(alpha=.2, linewidth=.5)
    save(fig, "figure3_discrimination_curves.pdf")

    fig, axes = plt.subplots(2, 3, figsize=(8.1, 5.8))
    for ax, model in zip(axes.flat, MODELS):
        y, pred, _ = prediction(model)
        cm = confusion_matrix(y, pred, labels=classes, normalize="true")
        draw_confusion(ax, cm, model.replace("_", " ").upper())
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.91, wspace=0.30, hspace=0.60)
    save(fig, "figure6_confusion_matrix_top6.pdf")

    y, pred, _ = prediction("tcst_har")
    cm = confusion_matrix(y, pred, labels=classes, normalize="true")
    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    im = draw_confusion(ax, cm, "TCST-HAR")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, label="Recall")
    fig.subplots_adjust(left=0.16, right=0.91, bottom=0.21, top=0.91)
    save(fig, "figure5_confusion_matrix_tcst.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), gridspec_kw={"width_ratios": [1.35, 1]}, constrained_layout=True)
    x = np.arange(len(LABELS))
    for model in MODELS:
        yt, yp, _ = prediction(model)
        axes[0].plot(x, f1_score(yt, yp, labels=classes, average=None, zero_division=0) * 100, marker="o", ms=2.5, lw=1.2, color=COLORS[model], label=model.replace("_", " ").upper())
    axes[0].set_ylim(88, 100.5)
    axes[0].set_xticks(x, LABELS, rotation=52, ha="right", fontsize=6)
    axes[0].set_ylabel("Class F1 (%)")
    axes[0].legend(fontsize=5.5, frameon=False, ncol=2, loc="lower left")
    axes[0].grid(axis="y", alpha=.2)
    error = confusion_matrix(y, pred, labels=classes).sum(1) - np.diag(confusion_matrix(y, pred, labels=classes))
    axes[1].barh(np.arange(len(LABELS)), error, color="#B2182B")
    axes[1].set_yticks(np.arange(len(LABELS)), LABELS, fontsize=6)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Misclassified sequences")
    axes[1].grid(axis="x", alpha=.2)
    for i, value in enumerate(error):
        axes[1].text(value + .05, i, str(int(value)), va="center", fontsize=6)
    save(fig, "figure2_class_error_map.pdf")

    _, rpred, _ = prediction("resnet1d")
    rcm = confusion_matrix(y, rpred, labels=classes, normalize="true")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3), constrained_layout=True)
    im = axes[0].imshow(cm, vmin=0, vmax=1, cmap="Blues")
    axes[0].set_title("TCST-HAR normalized confusion")
    diff = (cm - rcm) * 100
    im2 = axes[1].imshow(diff, vmin=-4, vmax=4, cmap="RdBu_r")
    axes[1].set_title("Recall change vs. ResNet1D (pp)")
    for ax in axes:
        ax.set_xticks(classes, LABELS, rotation=55, ha="right", fontsize=6)
        ax.set_yticks(classes, LABELS, fontsize=6)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.colorbar(im, ax=axes[0], fraction=.045, pad=.02)
    fig.colorbar(im2, ax=axes[1], fraction=.045, pad=.02)
    save(fig, "figure4_tcst_error_signature.pdf")

    raw_cm = confusion_matrix(y, pred, labels=classes)
    links = []
    for actual in classes:
        for predicted in classes:
            count = int(raw_cm[actual, predicted])
            if actual != predicted and count:
                links.append((LABELS[actual], LABELS[predicted], count))
    fig, ax = plt.subplots(figsize=(7.1, 2.9), constrained_layout=True)
    ax.axis("off")
    ax.set_title("Residual TCST-HAR error flows (3 sequences)", loc="left", weight="bold")
    left = sorted({source for source, _, _ in links})
    right = sorted({target for _, target, _ in links})
    yleft = np.linspace(.15, .85, len(left))
    yright = np.linspace(.15, .85, len(right))
    lp, rp = dict(zip(left, yleft)), dict(zip(right, yright))
    for source, target, count in links:
        ax.plot([.16, .84], [lp[source], rp[target]], lw=1.5 + count, alpha=.55, color="#B2182B")
        ax.text(.13, lp[source], source, ha="right", va="center", fontsize=7)
        ax.text(.87, rp[target], target, ha="left", va="center", fontsize=7)
        ax.text(.5, (lp[source] + rp[target]) / 2, str(count), ha="center", va="center", fontsize=7, color="#7F0000")
    save(fig, "figure7_error_flow_sankey.pdf")


if __name__ == "__main__":
    main()
