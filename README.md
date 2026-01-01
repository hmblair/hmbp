# hmbp

A simple matplotlib wrapper with consistent, publication-ready styling.

## Installation

```bash
pip install -e .
```

## Style

- Font: Helvetica
- Colormap: RdPu (primary), PiYG (secondary/diverging)
- 400 DPI output
- Consistent sizing: title 15pt, labels 14pt, ticks 13pt, legend 12pt

## Usage

```python
import hmbp

fig, ax = hmbp.new_figure()
hmbp.line_plot(y_values, x_values, label="Model A")
hmbp.set_labels("Title", "X Label", "Y Label")
hmbp.save("output.png")
```

## Available Functions

| Function | Description |
|----------|-------------|
| `line_plot` | Line with optional fill |
| `scatter_plot` | Scatter with optional color mapping |
| `histogram` | Color-mapped histogram |
| `bar_plot` | Vertical/horizontal bars |
| `box_plot` | Box plot distributions |
| `violin_plot` | Violin plot distributions |
| `heatmap` | 2D heatmap with colorbar |
| `line_plot_with_error` | Line with shaded error region |
| `confusion_matrix` | Annotated confusion matrix |
| `roc_curve` | ROC curve with AUC |
| `precision_recall_curve` | PR curve with AP |
| `residual_plot` | Regression residuals |
| `learning_curve` | Train/val learning curves |
| `metric_comparison` | Horizontal bar comparison |
| `volcano_plot` | Volcano plot for differential analysis |

## Helpers

- `new_figure(figsize)` - Create figure and axes
- `set_labels(title, xlabel, ylabel)` - Apply labels
- `save(path, fig, close)` - Save with auto-legend

## Quick API

Single-call functions that create, label, and save in one step:

```python
import hmbp

hmbp.quick_histogram(data, title="Scores", xlabel="Value", path="hist.png")
hmbp.quick_bar(values, labels, title="Comparison", ylabel="F1", path="bars.png")
hmbp.quick_confusion_matrix(cm, class_names=["A", "B"], path="cm.png")
```

Available: `quick_line`, `quick_scatter`, `quick_histogram`, `quick_bar`, `quick_heatmap`, `quick_confusion_matrix`, `quick_roc`, `quick_volcano`
