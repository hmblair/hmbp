"""
PGFPlots figure generation module.

Generates publication-ready LaTeX figures using PGFPlots,
with an API mirroring the matplotlib module.
"""

import re
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Union, Sequence
import numpy as np


# =============================================================================
# Document Template
# =============================================================================

PREAMBLE = r"""\documentclass[border=0.2cm]{standalone}

\usepackage{pgfplots}
\usetikzlibrary{patterns.meta}
\usepackage[scaled]{helvet}
\renewcommand\familydefault{\sfdefault}
\usepackage[T1]{fontenc}
\pgfplotsset{compat=1.18}

\begin{document}
"""

POSTAMBLE = r"""
\end{document}
"""

# =============================================================================
# Colors (matching reference style)
# =============================================================================

# Named colors from reference files
COLORS = {
    'red': 'F9665E',
    'blue': '799FCB',
    'gold': 'F1A226',
    'purple': 'A44694',
    'orange': 'FF9F43',
}

# Categorical colors (matching matplotlib module)
CATEGORICAL_COLORS = [
    'E41A1C',  # red
    '377EB8',  # blue
    '4DAF4A',  # green
    '984EA3',  # purple
    'FF7F00',  # orange
    'A65628',  # brown
    'F781BF',  # pink
    '999999',  # gray
]


def _color_definitions() -> str:
    """Generate LaTeX color definition commands."""
    lines = []
    for name, hex_val in COLORS.items():
        lines.append(rf"\definecolor{{{name}1}}{{HTML}}{{{hex_val}}}")
    for i, hex_val in enumerate(CATEGORICAL_COLORS):
        lines.append(rf"\definecolor{{cat{i}}}{{HTML}}{{{hex_val}}}")
    return '\n'.join(lines)


def _get_color(color: Optional[Union[str, int]], index: int = 0) -> str:
    """Get a color name for use in PGFPlots commands."""
    if color is None:
        return f"cat{index % len(CATEGORICAL_COLORS)}"
    if isinstance(color, int):
        return f"cat{color % len(CATEGORICAL_COLORS)}"
    if color in COLORS:
        return f"{color}1"
    # Assume it's a raw color specification
    return color


# =============================================================================
# Figure State
# =============================================================================

class _AxisState:
    """Holds state for a single axis environment."""
    def __init__(self, **options):
        self.plots: list = []
        self.title: str = ""
        self.xlabel: str = ""
        self.ylabel: str = ""
        self.options: dict = {}
        self.legend_entries: list = []
        self._plot_index: int = 0
        self.options.update(options)

    def next_color_index(self) -> int:
        idx = self._plot_index
        self._plot_index += 1
        return idx


class _FigureState:
    """Holds the current figure state (one tikzpicture, multiple axes)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self._axes: list = []
        self._current: Optional[_AxisState] = None

    @property
    def current(self) -> _AxisState:
        """Return the current axis, creating one if needed."""
        if self._current is None:
            self._current = _AxisState()
        return self._current

    def new_axis(self, **options) -> _AxisState:
        """Finalize any current axis and start a new one."""
        if self._current is not None:
            self._axes.append(self._current)
        self._current = _AxisState(**options)
        return self._current

    def all_axes(self) -> list:
        """Return all axes including the current one."""
        result = list(self._axes)
        if self._current is not None:
            result.append(self._current)
        return result


_state = _FigureState()


# =============================================================================
# Internal Helpers
# =============================================================================

def _format_coordinates(x: np.ndarray, y: np.ndarray) -> str:
    """Format x,y data as PGFPlots coordinates."""
    lines = []
    for xi, yi in zip(x, y):
        lines.append(f"\t({xi}, {yi})")
    return '\n'.join(lines)


def _parse_dimension(dim: str) -> tuple:
    """Parse a TeX dimension like '8pt' into (value, unit)."""
    m = re.match(r'([0-9.]+)\s*([a-z]*)', dim)
    if m:
        return float(m.group(1)), m.group(2) or 'pt'
    return float(dim), 'pt'


def _build_axis_options(axis: _AxisState) -> str:
    """Build the axis options string for a given axis."""
    opts = []

    if axis.title:
        opts.append(rf"title={{\large {axis.title}}}")
    if axis.xlabel:
        opts.append(rf"xlabel={{\large {axis.xlabel}}}")
    if axis.ylabel:
        opts.append(rf"ylabel={{\large {axis.ylabel}}}")

    opts.append("ymajorgrids")

    if 'axis lines' not in axis.options:
        opts.extend([
            "xtick pos=left",
            "ytick pos=left",
            "x axis line style=-",
            "y axis line style=-",
        ])

    for key, val in axis.options.items():
        if val is True:
            opts.append(key)
        elif val is not False and val is not None:
            opts.append(f"{key}={val}")

    if axis.legend_entries:
        opts.append("legend pos=north west")

    return ',\n\t'.join(opts)


def _build_document() -> str:
    """Build the complete LaTeX document."""
    axes = _state.all_axes()

    parts = [
        PREAMBLE.strip(),
        "",
        _color_definitions(),
        "",
        r"\begin{tikzpicture}",
    ]

    for axis in axes:
        parts.append("")
        parts.append(r"\begin{axis}[")
        parts.append(f"\t{_build_axis_options(axis)}")
        parts.append("]")
        parts.append("")
        parts.extend(axis.plots)
        for entry in axis.legend_entries:
            parts.append(rf"\addlegendentry{{{entry}}}")
        parts.append("")
        parts.append(r"\end{axis}")

    parts.extend([
        "",
        r"\end{tikzpicture}",
        POSTAMBLE.strip(),
    ])

    return '\n'.join(parts)


def _compile_tex(tex_path: Path) -> Optional[Path]:
    """Compile .tex to .pdf using lualatex or pdflatex."""
    tex_path = Path(tex_path)
    pdf_path = tex_path.with_suffix('.pdf')

    # Remove existing PDF to ensure fresh compilation
    if pdf_path.exists():
        pdf_path.unlink()

    # Try lualatex first (better Helvetica support), then pdflatex
    for compiler in ['lualatex', 'pdflatex']:
        if shutil.which(compiler) is None:
            continue

        try:
            subprocess.run(
                [compiler, '-interaction=nonstopmode', tex_path.name],
                cwd=tex_path.parent,
                capture_output=True,
                timeout=60,
            )

            # Check if PDF was created (may succeed even with warnings)
            if pdf_path.exists():
                # Clean up auxiliary files
                for ext in ['.aux', '.log']:
                    aux_file = tex_path.with_suffix(ext)
                    if aux_file.exists():
                        aux_file.unlink()
                return pdf_path

        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            continue

    return None


# =============================================================================
# Public API
# =============================================================================

def new_figure(**options) -> None:
    """
    Start a new figure context.

    Parameters
    ----------
    **options : dict
        Axis options for the first axis (e.g., ymode='log', xmin=0).
    """
    _state.reset()
    if options:
        _state.current.options.update(options)


def new_axis(**options) -> None:
    """
    Start a new axis within the current figure.

    Closes the current axis (if any) and opens a new one.
    All subsequent plot commands target this axis.

    Parameters
    ----------
    **options : dict
        PGFPlots axis options (e.g., xshift='193pt', ymode='log').
    """
    _state.new_axis(**options)


def line_plot(
    y: Sequence,
    x: Optional[Sequence] = None,
    label: str = "",
    color: Optional[Union[str, int]] = None,
    marker: str = "*",
    thick: bool = True,
) -> None:
    """
    Add a line plot to the current figure.

    Parameters
    ----------
    y : array-like
        The y-values to plot.
    x : array-like, optional
        The x-values. If None, uses indices.
    label : str, optional
        Label for the legend.
    color : str or int, optional
        Color name ('red', 'blue', etc.) or categorical index.
    marker : str, optional
        Marker style ('*', 'square*', 'o', 'none'). Default is '*'.
    thick : bool, optional
        Whether to use thick lines. Default is True.
    """
    y = np.asarray(y)
    if x is None:
        x = np.arange(len(y))
    else:
        x = np.asarray(x)

    color_name = _get_color(color, _state.current.next_color_index())

    opts = [f"color={color_name}"]
    if marker and marker != 'none':
        opts.append(f"mark={marker}")
    if thick:
        opts.append("thick")

    opts_str = ', '.join(opts)
    coords = _format_coordinates(x, y)

    plot_cmd = rf"\addplot [{opts_str}] coordinates {{{chr(10)}{coords}{chr(10)}}};"
    _state.current.plots.append(plot_cmd)

    if label:
        _state.current.legend_entries.append(label)


def multi_line_plot(
    ys: Sequence[Sequence],
    x: Optional[Sequence] = None,
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[Union[str, int]]] = None,
    marker: str = "*",
    thick: bool = True,
) -> None:
    """
    Add multiple line plots to the current figure.

    Parameters
    ----------
    ys : sequence of array-like
        List of y-value arrays to plot.
    x : array-like, optional
        Shared x-values. If None, uses indices.
    labels : sequence of str, optional
        Labels for each line.
    colors : sequence of str/int, optional
        Colors for each line.
    marker : str, optional
        Marker style. Default is '*'.
    thick : bool, optional
        Whether to use thick lines. Default is True.
    """
    n = len(ys)
    if labels is None:
        labels = [None] * n
    if colors is None:
        colors = [None] * n

    for y, label, color in zip(ys, labels, colors):
        line_plot(y, x=x, label=label or "", color=color, marker=marker, thick=thick)


def bar_plot(
    values: Sequence,
    labels: Sequence[str],
    color: Optional[Union[str, int]] = None,
    bar_width: str = "15pt",
    hatch: bool = False,
) -> None:
    """
    Add a bar plot to the current figure.

    Parameters
    ----------
    values : array-like
        The bar heights.
    labels : sequence of str
        Labels for each bar.
    color : str or int, optional
        Bar fill color.
    bar_width : str, optional
        Width of bars. Default is "15pt".
    hatch : bool, optional
        Whether to apply diagonal line hatching. Default is False.
    """
    values = np.asarray(values)
    n = len(values)
    color_name = _get_color(color, 0)
    axis = _state.current

    axis.options['ybar'] = True
    axis.options['bar width'] = bar_width
    axis.options['x'] = '0.7cm'
    axis.options['enlarge x limits'] = '{abs=0.5}'

    tick_positions = ', '.join(str(i) for i in range(n))
    tick_labels = ', '.join(labels)
    axis.options['xtick'] = f'{{{tick_positions}}}'
    axis.options['xticklabels'] = f'{{{tick_labels}}}'

    coords = []
    for i, val in enumerate(values):
        coords.append(f"\t({i}, {val})")
    coords_str = '\n'.join(coords)

    opts = [f"fill={color_name}"]
    if hatch:
        opts.append(r"postaction={pattern={Lines[angle=45, distance=2pt]}}")
    opts_str = ', '.join(opts)

    plot_cmd = rf"\addplot [{opts_str}] coordinates {{{chr(10)}{coords_str}{chr(10)}}};"
    axis.plots.append(plot_cmd)


def grouped_bar_plot(
    datasets: Sequence[Sequence],
    labels: Optional[Sequence[str]] = None,
    group_labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[Union[str, int]]] = None,
    hatch: Optional[Sequence[bool]] = None,
    bar_width: str = "8pt",
    group_spacing: str = "1cm",
) -> None:
    """
    Add a grouped bar plot for comparing multiple series.

    Parameters
    ----------
    datasets : sequence of array-like
        One array of values per series. All must have the same length.
    labels : sequence of str, optional
        Legend label for each series.
    group_labels : sequence of str, optional
        Tick label for each group (x-axis position).
    colors : sequence of str/int, optional
        Color for each series.
    hatch : sequence of bool, optional
        Whether to apply diagonal line hatching per series.
    bar_width : str, optional
        Width of each bar (TeX dimension). Default is "8pt".
    group_spacing : str, optional
        Spacing between group centers. Default is "1cm".
    """
    axis = _state.current
    datasets = [np.asarray(d) for d in datasets]
    n_series = len(datasets)
    n_groups = len(datasets[0])

    if labels is None:
        labels = [None] * n_series
    if colors is None:
        colors = [None] * n_series
    if hatch is None:
        hatch = [False] * n_series
    if group_labels is None:
        group_labels = [str(i) for i in range(n_groups)]

    axis.options['ybar'] = True
    axis.options['bar width'] = bar_width
    axis.options['x'] = group_spacing
    axis.options['enlarge x limits'] = '{abs=0.5}'

    tick_positions = ', '.join(str(i) for i in range(n_groups))
    tick_labels = ', '.join(group_labels)
    axis.options['xtick'] = f'{{{tick_positions}}}'
    axis.options['xticklabels'] = f'{{{tick_labels}}}'

    # Compute bar shifts to center the group
    bar_val, bar_unit = _parse_dimension(bar_width)
    total = n_series * bar_val
    shifts = [
        (i * bar_val) - (total - bar_val) / 2
        for i in range(n_series)
    ]

    for i, (values, label, color, hatched) in enumerate(
        zip(datasets, labels, colors, hatch)
    ):
        color_name = _get_color(color, axis.next_color_index())
        opts = [f"fill={color_name}", f"bar shift={shifts[i]}{bar_unit}"]
        if hatched:
            opts.append(
                r"postaction={pattern={Lines[angle=45, distance=2pt]}}"
            )

        opts_str = ', '.join(opts)
        coords = []
        for j, val in enumerate(values):
            coords.append(f"\t({j}, {val})")
        coords_str = '\n'.join(coords)

        plot_cmd = rf"\addplot [{opts_str}] coordinates {{{chr(10)}{coords_str}{chr(10)}}};"
        axis.plots.append(plot_cmd)

        if label:
            axis.legend_entries.append(label)


def histogram(
    data: Sequence,
    bins: int = 10,
    color: Optional[Union[str, int]] = None,
    density: bool = False,
) -> None:
    """
    Add a histogram to the current figure.

    Parameters
    ----------
    data : array-like
        The data to bin.
    bins : int, optional
        Number of bins. Default is 10.
    color : str or int, optional
        Bar fill color.
    density : bool, optional
        If True, normalize to density. Default is False.
    """
    data = np.asarray(data)
    counts, bin_edges = np.histogram(data, bins=bins, density=density)

    color_name = _get_color(color, 0)
    axis = _state.current

    axis.options['axis lines'] = 'left'
    axis.options['enlarge x limits'] = '0.05'
    axis.options['ymax'] = float(counts.max()) * 1.1

    coords = []
    for i, count in enumerate(counts):
        coords.append(f"({bin_edges[i]},{count})")
    coords.append(f"({bin_edges[-1]},{0})")
    coords_str = '\n'.join(coords)

    plot_cmd = rf"\addplot+[ybar interval, mark=no, fill={color_name}!50, draw=black, thin] plot coordinates{{{chr(10)}{coords_str}{chr(10)}}};"
    axis.plots.append(plot_cmd)


def node(
    x: float,
    y: float,
    text: str,
    anchor: str = "south",
    **options,
) -> None:
    """
    Add a text node at axis coordinates.

    Parameters
    ----------
    x : float
        X coordinate (in axis units).
    y : float
        Y coordinate (in axis units).
    text : str
        The text content (can include LaTeX).
    anchor : str, optional
        Node anchor point. Default is "south".
    **options : dict
        Additional TikZ node options. Underscores in keys are
        converted to spaces (e.g., align_center -> align center,
        minimum_size -> minimum size).
    """
    axis = _state.current
    opts = [f"anchor={anchor}"]
    for key, val in options.items():
        key = key.replace('_', ' ')
        if val is True:
            opts.append(key)
        elif val is not False and val is not None:
            opts.append(f"{key}={val}")
    opts_str = ', '.join(opts)
    cmd = rf"\node [{opts_str}] at (axis cs:{x},{y}) {{{text}}};"
    axis.plots.append(cmd)


def raw(latex: str) -> None:
    """
    Append raw LaTeX to the current axis.

    Use this for features not covered by the API.

    Parameters
    ----------
    latex : str
        Raw LaTeX code to insert into the current axis environment.
    """
    _state.current.plots.append(latex)


def set_labels(
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    """
    Set title and axis labels.

    Parameters
    ----------
    title : str, optional
        Figure title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    """
    if title:
        _state.current.title = title
    if xlabel:
        _state.current.xlabel = xlabel
    if ylabel:
        _state.current.ylabel = ylabel


def set_options(**options) -> None:
    """
    Set additional axis options.

    Parameters
    ----------
    **options : dict
        PGFPlots axis options (e.g., ymode='log', ymin=0).
    """
    _state.current.options.update(options)


def save(
    path: Union[str, Path],
    compile: bool = True,
) -> Optional[Path]:
    """
    Save the figure as .tex and optionally compile to PDF.

    Parameters
    ----------
    path : str or Path
        Output path (should end with .tex).
    compile : bool, optional
        Whether to compile to PDF. Default is True.

    Returns
    -------
    pdf_path : Path or None
        Path to compiled PDF if successful, None otherwise.
    """
    path = Path(path)
    if path.suffix != '.tex':
        path = path.with_suffix('.tex')

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Generate and write LaTeX
    content = _build_document()
    path.write_text(content)

    # Compile if requested
    pdf_path = None
    if compile:
        pdf_path = _compile_tex(path)

    return pdf_path


# =============================================================================
# Quick API
# =============================================================================

def _quick(plot_fn, args, kwargs, title, xlabel, ylabel, path, fig_options):
    """Helper for quick_* functions."""
    new_figure(**fig_options)
    plot_fn(*args, **kwargs)
    set_labels(title, xlabel, ylabel)
    if path:
        return save(path)
    return None


def quick_line(
    y,
    x=None,
    title="",
    xlabel="",
    ylabel="",
    path=None,
    **kw
):
    """Create a line plot in one call. Saves to path if provided."""
    return _quick(line_plot, (y,), {'x': x, **kw}, title, xlabel, ylabel, path, {})


def quick_lines(
    ys,
    x=None,
    labels=None,
    title="",
    xlabel="",
    ylabel="",
    path=None,
    **kw
):
    """Create a multi-line plot in one call. Saves to path if provided."""
    return _quick(multi_line_plot, (ys,), {'x': x, 'labels': labels, **kw}, title, xlabel, ylabel, path, {})


def quick_bar(
    values,
    labels,
    title="",
    xlabel="",
    ylabel="",
    path=None,
    **kw
):
    """Create a bar plot in one call. Saves to path if provided."""
    return _quick(bar_plot, (values, labels), kw, title, xlabel, ylabel, path, {})


def quick_grouped_bar(
    datasets,
    labels=None,
    group_labels=None,
    title="",
    xlabel="",
    ylabel="",
    path=None,
    **kw
):
    """Create a grouped bar plot in one call. Saves to path if provided."""
    return _quick(
        grouped_bar_plot, (datasets,),
        {'labels': labels, 'group_labels': group_labels, **kw},
        title, xlabel, ylabel, path, {},
    )


def quick_histogram(
    data,
    title="",
    xlabel="",
    ylabel="Count",
    path=None,
    **kw
):
    """Create a histogram in one call. Saves to path if provided."""
    return _quick(histogram, (data,), kw, title, xlabel, ylabel, path, {})
