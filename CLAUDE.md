# Claude Code Instructions for hmbp

This is a matplotlib plotting library with consistent, publication-ready styling.

## When to Create Visualizations

Create plots to document results whenever you:
- Run experiments or benchmarks
- Analyze data or compute statistics
- Compare models, methods, or configurations
- Generate metrics (accuracy, loss, timing, etc.)
- Process datasets and want to show distributions

## How to Use

Use the quick API for single-call plotting:

```python
import hmbp

# Line plots (training curves, time series)
hmbp.quick_line(y, x, title="Loss Curve", xlabel="Epoch", ylabel="Loss", path="figures/loss.png")

# Line plots with smoothing (for noisy training curves)
hmbp.quick_line(y, x, smooth=0.9, title="Smoothed Loss", path="figures/loss_smooth.png")

# Multi-line plots (comparing multiple series)
hmbp.quick_lines([y1, y2, y3], x, labels=["A", "B", "C"], title="Comparison", path="figures/lines.png")

# Multi-line with smoothing
hmbp.quick_lines([y1, y2], x, labels=["A", "B"], smooth=0.9, title="Smoothed", path="figures/lines_smooth.png")

# Scatter plots (correlations, embeddings)
hmbp.quick_scatter(x, y, title="Feature Correlation", xlabel="X", ylabel="Y", path="figures/scatter.png")

# Histograms (distributions)
hmbp.quick_histogram(data, title="Score Distribution", xlabel="Score", path="figures/hist.png")

# Overlay histograms (comparing distributions)
hmbp.quick_histogram_overlay([data1, data2], labels=["Before", "After"], title="Distribution Comparison", path="figures/hist_overlay.png")

# Bar plots (comparisons)
hmbp.quick_bar(values, labels, title="Model Comparison", ylabel="Accuracy", path="figures/bars.png")

# Heatmaps (matrices, correlations)
hmbp.quick_heatmap(matrix, title="Correlation Matrix", path="figures/heatmap.png")

# Confusion matrices
hmbp.quick_confusion_matrix(cm, class_names=["A", "B", "C"], path="figures/cm.png")

# ROC curves
hmbp.quick_roc(fpr, tpr, auc=0.95, title="ROC Curve", path="figures/roc.png")

# Volcano plots (differential analysis)
hmbp.quick_volcano(log_fc, pvalues, title="Differential Expression", path="figures/volcano.png")
```

## Guidelines

1. **Always save plots** - Use the `path` parameter to save to `figures/` or an appropriate directory
2. **Use descriptive titles** - Titles should explain what the plot shows
3. **Label axes** - Include units where applicable
4. **Create plots proactively** - Don't wait to be asked; visualize results as you generate them
5. **Use smoothing for noisy data** - Add `smooth=0.9` for training curves to reduce noise
6. **Multiple related plots** - When comparing multiple things, create separate plots or use the standard API with subplots

## Standard API (for complex plots)

For multi-series plots or subplots:

```python
import hmbp

fig, ax = hmbp.new_figure()
hmbp.line_plot(y1, x, label="Model A", ax=ax)
hmbp.line_plot(y2, x, label="Model B", cmap=hmbp.CMAP_ALT, ax=ax)
hmbp.set_labels("Comparison", "X", "Y", ax=ax)
hmbp.save("figures/comparison.png")
```

## Installation

```bash
pip install hmbp
```
