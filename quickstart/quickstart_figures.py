"""
Plotting utilities for the In Your Own Words quickstart.

Main entry point:
    themes_barchart(proportions, category_column, ...)

Helper:
    calculate_proportions(annotations_df, df, category_column)
"""

import textwrap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


# ── Sorting ───────────────────────────────────────────────────────────────────

def _sort_themes(proportions, sort_by, r2_dict=None, column_mapping=None):
    """Sort theme columns by the given method."""
    if sort_by == "majority":
        # Sort by the group with the highest average proportion (most dominant)
        majority_group = proportions.sum(axis=1).idxmax()
        return proportions.loc[:, proportions.loc[majority_group].sort_values(ascending=True).index]

    elif sort_by == "min_max_any_group":
        max_props = proportions.max(axis=0)
        return proportions.loc[:, max_props.sort_values(ascending=False).index]

    elif sort_by == "r2" and r2_dict:
        r2_values = {}
        for col in proportions.columns:
            original = column_mapping.get(col, col) if column_mapping else col
            r2_values[col] = r2_dict.get(original.replace("\n", " "), 0)
        return proportions.loc[:, sorted(proportions.columns, key=lambda x: r2_values.get(x, 0), reverse=True)]

    raise ValueError(f"Unknown sort_by: '{sort_by}'. Use 'majority', 'min_max_any_group', or 'r2'.")


# ── Proportions ───────────────────────────────────────────────────────────────

def calculate_proportions(annotations_df, df, category_column):
    """
    For each theme (column in annotations_df), compute the share of activations
    that come from each group in category_column.

    Parameters
    ----------
    annotations_df : pd.DataFrame
        Binary (0/1) DataFrame — rows = respondents, columns = themes.
    df : pd.DataFrame
        Survey DataFrame containing category_column.
    category_column : str
        Column name to group by (e.g. 'gender', 'race', any categorical column).

    Returns
    -------
    pd.DataFrame
        Rows = groups, columns = themes, values = proportion of activations.
    """
    combined = pd.concat(
        [annotations_df.reset_index(drop=True), df[[category_column]].reset_index(drop=True)],
        axis=1,
    )
    group_counts = combined.groupby(category_column).sum()
    return group_counts.div(group_counts.sum(axis=0), axis=1)


# ── Plotting ──────────────────────────────────────────────────────────────────

def themes_barchart(
    proportions,
    category_column,
    sort_by="min_max_any_group",
    color_map=None,
    label_map=None,
    category_order=None,
    r2_dict=None,
    n_activated=None,
    textwrap_width=100,
    figwidth=25,
    figheight=None,
    has_custom_bbox=None,
    no_legend=False,
):
    """
    Horizontal stacked bar chart showing theme proportions across groups.

    Parameters
    ----------
    proportions : pd.DataFrame
        Output of calculate_proportions() — rows = groups, columns = themes.
    category_column : str
        Used as the legend title (e.g. 'Gender', 'Race').
    sort_by : str
        How to order themes along the y-axis:
        - 'majority'          sort by the most common group's proportion
        - 'min_max_any_group' sort so the most group-specific themes come first
        - 'r2'                sort by r2_dict values (requires r2_dict)
    color_map : dict, optional
        {group_name: hex_color}. Auto-generated from a colorblind palette if None.
    label_map : dict, optional
        {group_name: display_label} for legend text.
    category_order : list, optional
        Order of groups in the stacked bars. Defaults to index order.
    r2_dict : dict, optional
        {theme_name: r2_value} used when sort_by='r2'.
    n_activated : pd.Series, optional
        Activation counts per theme, appended to y-axis labels.
    figwidth, figheight : float, optional
        Figure dimensions in inches.
    has_custom_bbox : tuple, optional
        bbox_to_anchor for the legend (e.g. (1, 1)).
    no_legend : bool
        If True, suppress the legend.
    """
    # ── Column wrapping ───────────────────────────────────────────────────────
    if n_activated is not None:
        wrapped, col_map = [], {}
        for col in proportions.columns:
            w = textwrap.fill(f"{col} (n={n_activated[col]})", width=textwrap_width)
            wrapped.append(w)
            col_map[w] = col
    else:
        wrapped = [textwrap.fill(col, width=textwrap_width) for col in proportions.columns]
        col_map = {w: o for w, o in zip(wrapped, proportions.columns)}
    proportions = proportions.copy()
    proportions.columns = wrapped

    # ── Sort ──────────────────────────────────────────────────────────────────
    proportions = _sort_themes(proportions, sort_by, r2_dict, col_map)

    # ── Category ordering + renormalize ───────────────────────────────────────
    if category_order is not None:
        category_order = [c for c in category_order if c in proportions.index]
    else:
        category_order = list(proportions.index)
    proportions = proportions.loc[category_order]
    col_sums = proportions.sum(axis=0).replace(0, np.nan)
    proportions = proportions.div(col_sums, axis=1).fillna(0)

    # ── Figure ────────────────────────────────────────────────────────────────
    num_themes = len(proportions.columns)
    width  = figwidth or 25
    height = figheight or max(12, 12 + (num_themes - 10) * 0.2)
    fig, ax = plt.subplots(figsize=(width, height))

    plt.rcParams.update({
        'font.size': 20, 'axes.labelsize': 18, 'axes.titlesize': 18,
        'xtick.labelsize': 18, 'ytick.labelsize': 22,
        'legend.fontsize': 18, 'legend.title_fontsize': 18,
    })

    # ── Colors ────────────────────────────────────────────────────────────────
    if color_map is None:
        palette = cm.get_cmap('tab10', len(category_order))
        color_map = {cat: mcolors.to_hex(palette(i)) for i, cat in enumerate(category_order)}

    colors = [color_map.get(cat, "#ACACAC") for cat in proportions.index]

    # ── Plot ──────────────────────────────────────────────────────────────────
    (proportions * 100).T.plot(kind='barh', stacked=True, ax=ax, color=colors)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("")
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    # ── Legend ────────────────────────────────────────────────────────────────
    if not no_legend:
        handles = [
            patches.Patch(
                color=color_map.get(cat, "#ACACAC"),
                label=(label_map or {}).get(cat, cat),
            )
            for cat in category_order
        ]
        legend = ax.legend(
            handles=handles,
            title=category_column.replace("_", " ").title(),
            bbox_to_anchor=has_custom_bbox,
            loc='best',
            frameon=True, framealpha=0.9,
            facecolor='white', edgecolor='lightgray',
        )
        legend.get_title().set_fontweight('bold')

    # ── Alternating row backgrounds ───────────────────────────────────────────
    for i in range(num_themes):
        if i % 2 == 0:
            ax.axhspan(i - 0.4, i + 0.4, color='#f5f5f5', zorder=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig
