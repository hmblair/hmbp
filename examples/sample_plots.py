"""Generate sample plots to demonstrate the hmbp plotting module."""

import shutil
from itertools import count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import hmbp

np.random.seed(42)

FIGURES_DIR = Path("figures")
shutil.rmtree(FIGURES_DIR, ignore_errors=True)
FIGURES_DIR.mkdir(exist_ok=True)

_counter = count(1)


def savefig(name: str, fig=None):
    """Save with auto-incrementing number prefix."""
    path = FIGURES_DIR / f"{next(_counter):02d}_{name}.png"
    hmbp.save(str(path), fig)


# --- Core API examples ---

# Line plot
fig, ax = hmbp.new_figure()
x = np.linspace(0, 10, 100)
hmbp.line_plot(np.sin(x), x, label="Model A")
hmbp.line_plot(np.cos(x), x, label="Model B", cmap=hmbp.CMAP_ALT)
hmbp.set_labels("Training Loss Over Time", "Epoch", "Loss")
savefig("line_plot")

# Scatter plot
fig, ax = hmbp.new_figure()
x = np.random.randn(200)
y = 0.5 * x + np.random.randn(200) * 0.3
hmbp.scatter_plot(x, y, c=y, label="Samples")
hmbp.set_labels("Feature Correlation", "Feature A", "Feature B")
savefig("scatter_plot")

# Histogram
fig, ax = hmbp.new_figure()
data = np.concatenate([np.random.randn(500), np.random.randn(300) + 3])
hmbp.histogram(data, bins=40)
hmbp.set_labels("Score Distribution", "Score", "Count")
savefig("histogram")

# Bar plot
fig, ax = hmbp.new_figure()
models = ["Random Forest", "XGBoost", "Neural Net", "SVM", "LogReg"]
scores = [0.92, 0.95, 0.91, 0.88, 0.85]
hmbp.bar_plot(scores, models)
hmbp.set_labels("Model Comparison", "", "F1 Score")
savefig("bar_plot")

# Grouped bar plot
fig, ax = hmbp.new_figure()
hmbp.grouped_bar_plot(
    [[0.92, 0.88, 0.95], [0.89, 0.91, 0.93]],
    labels=["Train", "Validation"],
    group_labels=["Random Forest", "XGBoost", "Neural Net"],
)
hmbp.set_labels("Train vs Validation Performance", "", "F1 Score")
savefig("grouped_bar_plot")

# Box plot
fig, ax = hmbp.new_figure()
data = [np.random.randn(100) + i * 0.5 for i in range(4)]
hmbp.box_plot(data, ["Model A", "Model B", "Model C", "Model D"])
hmbp.set_labels("Score Distribution by Model", "", "Score")
savefig("box_plot")

# Violin plot
fig, ax = hmbp.new_figure()
data = [np.random.randn(100) * (i + 1) * 0.3 for i in range(4)]
hmbp.violin_plot(data, ["Small", "Medium", "Large", "XL"])
hmbp.set_labels("Prediction Variance by Model Size", "", "Prediction Error")
savefig("violin_plot")

# Heatmap (correlation matrix)
fig, ax = hmbp.new_figure()
corr = np.random.randn(6, 6)
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)
features = ["feat_1", "feat_2", "feat_3", "feat_4", "feat_5", "feat_6"]
hmbp.heatmap(corr, xticklabels=features, yticklabels=features,
             colorbar_label="Correlation", center_zero=True, annot=True)
hmbp.set_labels("Feature Correlation Matrix", "", "")
savefig("heatmap")

# Line plot with error
fig, ax = hmbp.new_figure()
x = np.arange(10)
y = np.exp(-x * 0.3) + 0.1
yerr = 0.05 + 0.02 * np.random.randn(10)
hmbp.line_plot_with_error(y, np.abs(yerr), x, label="Mean +/- Std")
hmbp.set_labels("Convergence with Uncertainty", "Iteration", "Loss")
savefig("line_with_error")

# Confusion matrix
fig, ax = hmbp.new_figure()
cm = np.array([[85, 10, 5], [8, 82, 10], [4, 12, 84]])
hmbp.confusion_matrix(cm, class_names=["Cat", "Dog", "Bird"], normalize=True)
hmbp.set_labels("Classification Results", "", "")
savefig("confusion_matrix")

# ROC curve
fig, ax = hmbp.new_figure()
fpr1 = np.linspace(0, 1, 100)
tpr1 = np.sqrt(fpr1)
tpr2 = fpr1 ** 2 + fpr1 * 0.5
tpr2 = np.clip(tpr2, 0, 1)
hmbp.roc_curve(fpr1, tpr1, auc=0.92, label="XGBoost")
hmbp.roc_curve(fpr1, tpr2, auc=0.71, label="Baseline", cmap=hmbp.CMAP_ALT)
hmbp.set_labels("ROC Comparison", "", "")
savefig("roc_curve")

# Precision-Recall curve
fig, ax = hmbp.new_figure()
recall = np.linspace(0, 1, 100)
precision1 = 1 - 0.3 * recall ** 2
precision2 = 1 - 0.6 * recall
hmbp.precision_recall_curve(precision1, recall, ap=0.89, label="Model A")
hmbp.precision_recall_curve(precision2, recall, ap=0.72, label="Model B", cmap=hmbp.CMAP_ALT)
hmbp.set_labels("Precision-Recall Comparison", "", "")
savefig("pr_curve")

# Residual plot
fig, ax = hmbp.new_figure()
y_pred = np.linspace(0, 10, 100)
y_true = y_pred + np.random.randn(100) * 0.5
hmbp.residual_plot(y_true, y_pred)
hmbp.set_labels("Residual Analysis", "", "")
savefig("residual_plot")

# Learning curve
fig, ax = hmbp.new_figure()
sizes = np.array([100, 200, 500, 1000, 2000, 5000])
train_scores = np.column_stack([
    0.99 - 0.1 * np.exp(-sizes / 500) + np.random.randn(6) * 0.01
    for _ in range(5)
])
val_scores = np.column_stack([
    0.85 + 0.1 * (1 - np.exp(-sizes / 1000)) + np.random.randn(6) * 0.02
    for _ in range(5)
])
hmbp.learning_curve(train_scores, val_scores, sizes, metric_name="Accuracy")
hmbp.set_labels("Learning Curve", "", "")
savefig("learning_curve")

# Metric comparison
fig, ax = hmbp.new_figure()
metrics = {
    "Accuracy": 0.94,
    "Precision": 0.91,
    "Recall": 0.88,
    "F1": 0.89,
    "AUC": 0.96
}
hmbp.metric_comparison(metrics)
hmbp.set_labels("Model Metrics", "", "")
savefig("metric_comparison")

# Histogram overlay
fig, ax = hmbp.new_figure()
data1 = np.random.randn(500) * 0.8
data2 = np.random.randn(500) * 1.2 + 1.5
data3 = np.random.randn(500) * 0.6 + 3
hmbp.histogram_overlay([data1, data2, data3], labels=["MLP", "Attention", "Ground Truth"], bins=30)
hmbp.set_labels("Score Distributions by Model", "Score", "Count")
savefig("histogram_overlay")

# Multi-line plot with smoothing
fig, ax = hmbp.new_figure()
x = np.arange(100)
y1 = np.exp(-x * 0.03) + np.random.randn(100) * 0.08
y2 = np.exp(-x * 0.025) * 0.9 + np.random.randn(100) * 0.08
y3 = np.exp(-x * 0.035) * 1.1 + np.random.randn(100) * 0.08
hmbp.multi_line_plot([y1, y2, y3], x, labels=["Adam", "SGD", "AdamW"], smooth=0.9)
hmbp.set_labels("Training Loss Comparison (Smoothed)", "Epoch", "Loss")
savefig("multi_line")

# Raw vs smoothed comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(100)
y = np.exp(-x * 0.03) + np.random.randn(100) * 0.15
hmbp.line_plot(y, x, label="Raw", ax=ax1, fill=False)
hmbp.set_labels("Raw Training Curve", "Epoch", "Loss", ax=ax1)
hmbp.line_plot(y, x, label="Smoothed (0.9)", ax=ax2, fill=False, smooth=0.9)
hmbp.set_labels("EMA Smoothed (weight=0.9)", "Epoch", "Loss", ax=ax2)
savefig("smoothing_comparison")

# Scatter plot with regression line
fig, ax = hmbp.new_figure()
x = np.random.randn(100)
y = 1.5 * x + 0.5 + np.random.randn(100) * 0.8
hmbp.scatter_plot(x, y, regression=True)
hmbp.set_labels("Scatter with Regression", "X", "Y")
savefig("scatter_regression")

# =============================================================================
# Quick API examples - single-call convenience functions
# =============================================================================

# Quick line plot
hmbp.quick.line(
    np.sin(np.linspace(0, 10, 100)),
    np.linspace(0, 10, 100),
    title="Quick Line Plot",
    xlabel="X",
    ylabel="sin(x)",
    path=str(FIGURES_DIR / f"{next(_counter):02d}_quick_line.png"),
)

# Quick scatter plot
x = np.random.randn(100)
y = 0.8 * x + np.random.randn(100) * 0.3
hmbp.quick.scatter(x, y, title="Quick Scatter", xlabel="X", ylabel="Y",
                   path=str(FIGURES_DIR / f"{next(_counter):02d}_quick_scatter.png"))

# Quick histogram
hmbp.quick.histogram(
    np.random.randn(1000),
    title="Quick Histogram",
    xlabel="Value",
    path=str(FIGURES_DIR / f"{next(_counter):02d}_quick_histogram.png"),
)

# Quick multi-line with smoothing
x = np.arange(100)
y1 = np.exp(-x * 0.03) + np.random.randn(100) * 0.1
y2 = np.exp(-x * 0.025) * 0.9 + np.random.randn(100) * 0.1
hmbp.quick.lines(
    [y1, y2], x,
    labels=["Model A", "Model B"],
    smooth=0.9,
    title="Quick Lines (Smoothed)",
    xlabel="Epoch",
    ylabel="Loss",
    path=str(FIGURES_DIR / f"{next(_counter):02d}_quick_lines.png"),
)

# Quick bar plot
hmbp.quick.bar(
    [0.92, 0.88, 0.95, 0.84],
    ["RF", "SVM", "XGB", "LR"],
    title="Quick Bar",
    ylabel="F1 Score",
    path=str(FIGURES_DIR / f"{next(_counter):02d}_quick_bar.png"),
)

# Quick grouped bar plot
hmbp.quick.bars(
    [[0.92, 0.88, 0.95, 0.84], [0.90, 0.85, 0.93, 0.82]],
    labels=["Train", "Test"],
    group_labels=["RF", "SVM", "XGB", "LR"],
    title="Quick Grouped Bars",
    ylabel="F1 Score",
    path=str(FIGURES_DIR / f"{next(_counter):02d}_quick_bars.png"),
)

# Quick violin plot
data = [np.random.randn(100) * (i + 1) * 0.3 for i in range(3)]
hmbp.quick.violin(
    data,
    ["Small", "Medium", "Large"],
    title="Quick Violin",
    ylabel="Error",
    path=str(FIGURES_DIR / f"{next(_counter):02d}_quick_violin.png"),
)

n = next(_counter) - 1
print(f"Generated {n} sample plots in {FIGURES_DIR}/")
