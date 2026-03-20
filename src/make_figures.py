import textwrap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import pandas as pd
import matplotlib.ticker as mtick

def _get_sorted_columns(proportions, sort_by, r2_dict=None, column_mapping=None):
    """Helper function to handle different sorting methods."""
    if sort_by == "white_proportion":
        if "White" not in proportions.index:
            raise ValueError("'White' group not found in the data. Please check the input.")
        return proportions.loc[:, proportions.loc["White"].sort_values(ascending=True).index]
    
    elif sort_by == "cis_proportion":
        # Create a new column that combines Cisgender Man and Cisgender Woman proportions
        cis_proportions = pd.Series(index=proportions.columns, dtype=float)
        for col in proportions.columns:
            man_value = proportions.loc["Cisgender Man", col] if "Cisgender Man" in proportions.index else 0
            woman_value = proportions.loc["Cisgender Woman", col] if "Cisgender Woman" in proportions.index else 0
            cis_proportions[col] = man_value + woman_value
        return proportions.loc[:, cis_proportions.sort_values(ascending=True).index]
    
    elif sort_by == "cisman_ciswoman":
        '''First sort by cisman, then ciswoman'''
        # Create two series for sorting
        cisman_proportions = pd.Series(index=proportions.columns, dtype=float)
        ciswoman_proportions = pd.Series(index=proportions.columns, dtype=float)
        
        # Calculate proportions for each column
        for col in proportions.columns:
            cisman_proportions[col] = proportions.loc["Cisgender Man", col] if "Cisgender Man" in proportions.index else 0
            ciswoman_proportions[col] = proportions.loc["Cisgender Woman", col] if "Cisgender Woman" in proportions.index else 0
        
        # Create a DataFrame for sorting
        sort_df = pd.DataFrame({
            'cisman': cisman_proportions,
            'ciswoman': ciswoman_proportions
        })
        
        # Sort first by cisman, then by ciswoman
        sorted_cols = sort_df.sort_values(['cisman', 'ciswoman']).index
        
        return proportions.loc[:, sorted_cols]
    elif sort_by == "straight":
        group = "Straight or heterosexual"
        if group not in proportions.index:
            raise ValueError(f"'{group}' group not found in the data. Please check the input.")
        return proportions.loc[:, proportions.loc[group].sort_values(ascending=True).index]
    
    elif sort_by == "min_majority":
        # double check this actually works...
        balance_scores = pd.Series(index=proportions.columns, dtype=float)
        for col in proportions.columns:
            majority_prop = proportions[col].max()
            balance_scores[col] = min(1 - majority_prop, majority_prop)
        return proportions.loc[:, balance_scores.sort_values(ascending=True).index]
    elif sort_by == "min_max_any_group":
        # Sort by the minimum of the maximum proportion for each theme (column)
        max_props = proportions.max(axis=0)  # max for each column (theme)
        return proportions.loc[:, max_props.sort_values(ascending=False).index]
    elif sort_by == "r2" and r2_dict:
        # Use column_mapping to look up original column names for r2_dict
        r2_values = {}
        for col in proportions.columns:
            original_col = column_mapping.get(col, col) if column_mapping else col
            r2_values[col] = r2_dict.get(original_col.replace("\n", " "), 0)
        return proportions.loc[:, sorted(proportions.columns, key=lambda x: r2_values.get(x, 0), reverse=True)]
    
    raise ValueError(f"Sorting method {sort_by} not supported")

def _get_color_mapping(identity, custom_colors=False):
    """Get color mapping based on identity type."""
    print(f"Custom colors: {custom_colors}")
    if custom_colors:
        return {
        # race (YlGnBu - Darker shades emphasized and ordered dark-to-light)
        "White": "#41B6C4",
        # "White": "#081D58",
        "Black or African American": "#0C2C84",
        "Asian": "#225EA8",
        "Hispanic or Latino": "#1D91C0",
        "Middle Eastern or North African": "#41B6C4",
        "American Indian or Alaska Native": "#7FCDBB",
        "Native Hawaiian or Pacific Islander": "#C7E9B4",
        "Two or More Races": "#D7E897",
        "Some Other Race": "#081D58", 
        
# gender (Summer - Stretched to full Yellow brightness)
        "Cisgender Woman": "#008066",      # Deepest jungle green (Start)
        "Cisgender Man": "#339966",        # Strong medium green
        "Cisgender (Other)": "#66B266",    # Soft green
        "Transgender Woman": "#AACC66",    # Yellow-green (Distinct shift here)
        "Transgender Man": "#CCE666",      # Lime yellow
        "Transgender (Other)": "#EFEF81",  # Brightest yellow (End)
        "Prefer not to answer": "#C9C9C9", # Neutral grey anchor

        # sexual orientation (OrRd - Deep Red to Orange gradient)
        # "Straight or heterosexual": "#7F0000",  # Darkest red
        "Straight or heterosexual": "#F8DB6F",  # Darkest red
        "Gay or Lesbian": "#A30000",             # Deep crimson
        "Bisexual and/or Pansexual": "#C2140D",  # Bright red
        "Queer": "#DA3724",                      # Red-orange
        "Asexual or aromantic": "#EC5D42",       # Tomato
        "Sexually fluid": "#F67B51",             # Coral
        "Demisexual": "#FC9964",                 # Light coral
        "Questioning": "#FDBA83",                # Apricot
        "Multiple Identities": "#FDCD96",        # Peach
        "Other": "#7F0000",                      # Light peach
        "Prefer not to answer": "#C9C9C9",       # Neutral grey anchor

        }

    if identity == "race":
        return {
        "American Indian or Alaska Native": "#E78336",           
        "Asian": "#1B9E77",                  # Teal Green
        "Black or African American": "#7570B3",                  # Medium Purple
        "Hispanic or Latino": "#EB6D6D",     # Deep Pink
        "Middle Eastern or North African": "#66A61E",                   # Olive Green
        "Native Hawaiian or Pacific Islander": "#A6761D",                   # Mustard Yellow
        "Some Other Race": "#BEBEBE",        # Brown
        "Two or More Races": "#FCCE4F",      # Dark Gray E6AB02
        "White": "#1F78B4"                   # Rich Blue
        }

    elif identity == "gender":
        return {
        "Cisgender Woman": "#4878CF",          # Soft Blue
        "Cisgender Man": "#8DB4E2",            # Medium Blue
        "Cisgender (Other)": "#5A9CA5",        # Muted Teal 
        "Transgender Woman": "#E899B2",        # Soft Pink
        "Transgender Man": "#A25C9C",          # Plum
        "Transgender (Other)": "#C16E7D",      # Rosewood
        "Prefer not to answer": "#A0A0A0"      # Neutral Gray
        }

    elif identity == "sexual_orientation":
        return {
    "Straight or heterosexual": "#1F78B4",     # Same Rich Blue as White in race palette
    "Gay or Lesbian": "#7A5195",              # Deep Violet (distinct from gender's plum)
    "Bisexual and/or Pansexual": "#EF5675",   # Bold Rose (pairs with gender’s soft pink)
    "Queer": "#FFA600",                       # Golden Amber (warmer than MENA olive)
    "Asexual or aromantic": "#59B6C2",        # Slate Blue (distinct from White/race blue)
    "Sexually fluid": "#BC5090",              # Plum Rose (coheres with Trans/Two+Races)
    "Demisexual": "#EB6D6D",                  # Forest Sage (contrasts MENA olive)
    "Questioning": "#DAA520",                 # Neutral Light Gray
    "Multiple Identities": "#1B9E77",         # Burnt Sienna (earthy, shared with race tones)
    "Other": "#888888",                       # Mid Gray (generic fallback)
    "Prefer not to answer": "#A0A0A0",        # Very Light Gray
}
    return None

def _add_r2_values(ax, proportions, r2_dict, column_mapping=None):
    """Add R² values to the plot with gradient coloring."""
    if not r2_dict:
        return
        
    all_r2_values = []
    for interpretation in proportions.columns:
        original_col = column_mapping.get(interpretation, interpretation) if column_mapping else interpretation
        r2_val = r2_dict.get(original_col.replace("\n", " "), 0)
        all_r2_values.append(r2_val)
    
    max_r2 = max(all_r2_values)
    norm = mcolors.Normalize(vmin=0, vmax=max_r2)
    cmap = plt.cm.Greens
    
    for i, interpretation in enumerate(proportions.columns):
        original_col = column_mapping.get(interpretation, interpretation) if column_mapping else interpretation
        r2_value = r2_dict.get(original_col.replace("\n", " "), 0)
        
        if r2_value is not None:
            r2_formatted = f"R² = {r2_value:.2f}" if i == len(proportions.columns)-1 else f"{r2_value:.2f}"
            box_color = cmap(norm(r2_value))
            text_color = 'white' if norm(r2_value) > 0.5 else 'black'
            
            ax.text(
                1.01, i, r2_formatted,
                va='center', ha='left',
                fontsize=10, color=text_color,
                fontweight='bold',
                bbox=dict(
                    facecolor=box_color,
                    alpha=0.9,
                    boxstyle='round,pad=0.4',
                    edgecolor='none'
                )
            )

def _setup_plot_style():
    """Configure global plot styling."""
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 22,
        'legend.fontsize': 18,
        'legend.title_fontsize': 18
    })

def improved_race_themes_barchart(proportions, IDENTITY, sort_by, r2_dict=None, n_activated=None, custom_colors=False, custom_style=None, no_legend=False, textwrap_width=None, figwidth=None, figheight=None, has_custom_bbox=None):
    """
    Create a horizontal stacked bar chart showing theme proportions across identity groups.
    
    Parameters:
        proportions (pd.DataFrame): DataFrame with identity groups as rows and themes as columns
        IDENTITY (str): Type of identity being plotted ('race', 'gender', etc.)
        sort_by (str): Method to sort themes ('white_proportion', 'min_majority', 'r2')
        r2_dict (dict, optional): Dictionary mapping themes to R² values
        n_activated (pd.Series, optional): Series containing count of activations per theme
    """
    # Create a mapping between original and modified column names
    column_mapping = {}
    textwrap_width = textwrap_width if textwrap_width else 100
    if n_activated is not None:
        wrapped_columns = []
        for col in proportions.columns:
            modified_col = textwrap.fill(f"{col} (n={n_activated[col]})", width=textwrap_width)
            wrapped_columns.append(modified_col)
            column_mapping[modified_col] = col
    else:
        wrapped_columns = [textwrap.fill(col, width=textwrap_width) for col in proportions.columns]
        column_mapping = {wrapped: original for wrapped, original in zip(wrapped_columns, proportions.columns)}
    
    proportions.columns = wrapped_columns
    
    # Sort columns based on specified method
    proportions = _get_sorted_columns(proportions, sort_by, r2_dict, column_mapping)

    # Setup figure
    num_themes = len(proportions.columns)
    height = max(12, 12 + (num_themes - 10) * 0.2)
    width = figwidth if figwidth else 25
    height = figheight if figheight else height
    fig, ax = plt.subplots(figsize=(width, height))

    # Configure plot style
    custom_style() if custom_style else _setup_plot_style()
    
    # Customize category order for race
    if IDENTITY == "race":
        category_order = [
            "White",
            "Black or African American",
            "Asian",
            "Hispanic or Latino",
            "Middle Eastern or North African",
            "American Indian or Alaska Native",
            "Native Hawaiian or Pacific Islander",
            "Two or More Races",
            "Some Other Race",

        ]
        # Only keep categories present in the data
        category_order = [cat for cat in category_order if cat in proportions.index]
        proportions = proportions.loc[category_order]

    color_mapping = _get_color_mapping(IDENTITY, custom_colors)
    if color_mapping:
        colors = [color_mapping.get(category, "#ACACAC") for category in proportions.index]
    else:
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(proportions.index)))

    plot_df = proportions * 100
    plot_df.T.plot(kind='barh', stacked=True, ax=ax, color=colors)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("")
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add R² values if provided
    # _add_r2_values(ax, proportions, r2_dict, column_mapping)
    
    # Configure legend with custom order and labels
    legend_labels = [
        ("White", "White"),
        ("Black or African American", "Black"),
        ("Asian", "Asian"),
        ("Hispanic or Latino", "Latino"),
        ("Middle Eastern or North African", "MENA"),
        ("American Indian or Alaska Native", "AIAN"),
        ("Native Hawaiian or Pacific Islander", "NHPI"),
        ("Two or More Races", "2+ Races"),
        ("Some Other Race", "Other"),
    ]
    # Only keep those present
    legend_labels = [(cat, label) for cat, label in legend_labels if cat in proportions.index]
    handles = [patches.Patch(color=color_mapping.get(cat, "#ACACAC"), label=label) for cat, label in legend_labels]
    
    legend = ax.legend(
        handles=handles,
        title=f"{IDENTITY.capitalize()}",
        bbox_to_anchor=has_custom_bbox,
        loc='best',
        frameon=True,
        framealpha=0.9,
        facecolor='white',
        edgecolor='lightgray'
    )
    legend.get_title().set_fontweight('bold')
    
    # Add alternating row backgrounds
    for i in range(len(proportions.columns)):
        if i % 2 == 0:
            ax.axhspan(i-0.4, i+0.4, color='#f5f5f5', zorder=0)
    
    # Clean up plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if no_legend:
        plt.legend('', frameon=False)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def improved_gender_themes_barchart(proportions, IDENTITY, sort_by, r2_dict=None, n_activated=None, custom_colors=False, custom_style=None, no_legend=False, textwrap_width=None, figwidth=None, figheight=None,
                                    has_custom_bbox=None):
    """
    Create a horizontal stacked bar chart showing theme proportions across gender identity groups.
    
    Parameters:
        proportions (pd.DataFrame): DataFrame with identity groups as rows and themes as columns
        IDENTITY (str): Type of identity being plotted ('gender')
        sort_by (str): Method to sort themes ('cisman_proportion', 'min_majority', 'r2')
        r2_dict (dict, optional): Dictionary mapping themes to R² values
        n_activated (pd.Series, optional): Series containing count of activations per theme
    """
    # Create a mapping between original and modified column names
    column_mapping = {}
    textwrap_width = textwrap_width if textwrap_width else 100
    if n_activated is not None:
        wrapped_columns = []
        for col in proportions.columns:
            modified_col = textwrap.fill(f"{col} (n={n_activated[col]})", width=textwrap_width)
            wrapped_columns.append(modified_col)
            column_mapping[modified_col] = col
    else:
        wrapped_columns = [textwrap.fill(col, width=textwrap_width) for col in proportions.columns]
        column_mapping = {wrapped: original for wrapped, original in zip(wrapped_columns, proportions.columns)}
    
    proportions.columns = wrapped_columns
    
    # Sort columns based on specified method
    proportions = _get_sorted_columns(proportions, sort_by, r2_dict, column_mapping)

    # Setup figure
    num_themes = len(proportions.columns)
    # height = max(12, 12 + (num_themes - 10) * 0.25)
    height = max(12, 12 + (num_themes - 10) * 0.2)
    width = figwidth if figwidth else 25
    height = figheight if figheight else height

    fig, ax = plt.subplots(figsize=(width, height))

    # Configure plot style
    custom_style() if custom_style else _setup_plot_style()
    
    # Customize category order for gender
    if IDENTITY == "gender":
        category_order = [
            "Cisgender Woman",
            "Cisgender Man",
            "Cisgender (Other)",
            "Transgender Woman",
            "Transgender Man",
            "Transgender (Other)",
            "Prefer not to answer"
        ]
        category_order = [cat for cat in category_order if cat in proportions.index]
        proportions = proportions.loc[category_order]
    # Get colors
    color_mapping = _get_color_mapping(IDENTITY, custom_colors)
    print(color_mapping)
    if color_mapping:
        colors = [color_mapping.get(category, "#ACACAC") for category in proportions.index]
    else:
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(proportions.index)))
    # Create plot
    plot_df = proportions * 100
    plot_df.T.plot(kind='barh', stacked=True, ax=ax, color=colors)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
    
    # Add styling
    # ax.set_xlabel(f"Category Breakdown of Respondents Mentioning Each Theme", 
    #              fontsize=18, fontweight='bold', ha='left', x=-0.55)
    ax.set_ylabel("")
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # # Add R² values if provided
    # _add_r2_values(ax, proportions, r2_dict, column_mapping)
    
    # Configure legend with custom order and labels
    legend_labels = [
        ("Cisgender Woman", "Cis Woman"),
        ("Cisgender Man", "Cis Man"),
        ("Cisgender (Other)", "Cis Other"),
        ("Transgender Woman", "Trans Woman"),
        ("Transgender Man", "Trans Man"),
        ("Transgender (Other)", "Trans Other"),
        ("Prefer not to answer", "No Answer")
    ]
    legend_labels = [(cat, label) for cat, label in legend_labels if cat in proportions.index]
    handles = [patches.Patch(color=color_mapping.get(cat, "#ACACAC"), label=label) for cat, label in legend_labels]
    legend = ax.legend(
        handles=handles,
        title=f"{IDENTITY.capitalize()}",
        bbox_to_anchor=has_custom_bbox,
        loc='best',
        frameon=True,
        framealpha=0.9,
        facecolor='white',
        edgecolor='lightgray'
    )
    legend.get_title().set_fontweight('bold')
    
    # Add alternating row backgrounds
    for i in range(len(proportions.columns)):
        if i % 2 == 0:
            ax.axhspan(i-0.4, i+0.4, color='#f5f5f5', zorder=0)
    
    # Clean up plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if no_legend:
        plt.legend('', frameon=False) 
    plt.tight_layout()
    plt.show()
    
    return fig

def improved_sexual_orientation_themes_barchart(proportions, IDENTITY, sort_by, r2_dict=None, n_activated=None, custom_colors=False, custom_style=None, no_legend=False, textwrap_width=None, figwidth=None, figheight=None, has_custom_bbox=None):
    """
    Create a horizontal stacked bar chart showing theme proportions across sexual orientation groups.
    
    Parameters:
        proportions (pd.DataFrame): DataFrame with identity groups as rows and themes as columns
        IDENTITY (str): Type of identity being plotted ('sexual_orientation')
        sort_by (str): Method to sort themes ('straight', 'min_majority', 'r2')
        r2_dict (dict, optional): Dictionary mapping themes to R² values
        n_activated (pd.Series, optional): Series containing count of activations per theme
    """
    # Create a mapping between original and modified column names
    column_mapping = {}
    textwrap_width = textwrap_width if textwrap_width else 90
    if n_activated is not None:
        wrapped_columns = []
        for col in proportions.columns:
            modified_col = textwrap.fill(f"{col} (n={n_activated[col]})", width=textwrap_width)
            wrapped_columns.append(modified_col)
            column_mapping[modified_col] = col
    else:
        wrapped_columns = [textwrap.fill(col, width=textwrap_width) for col in proportions.columns]
        column_mapping = {wrapped: original for wrapped, original in zip(wrapped_columns, proportions.columns)}
    
    proportions.columns = wrapped_columns
    
    # Sort columns based on specified method
    proportions = _get_sorted_columns(proportions, sort_by, r2_dict, column_mapping)
    
    # Setup figure
    num_themes = len(proportions.columns)
    height = max(12, 12 + (num_themes - 10) * 0.2)
    width = figwidth if figwidth else 25
    height = figheight if figheight else height
    
    fig, ax = plt.subplots(figsize=(width, height))

    # height = max(12, 12 + (num_themes - 10) * 0.25)
    # fig, ax = plt.subplots(figsize=(22, height))
    
    # Configure plot style
    custom_style() if custom_style else _setup_plot_style()
    
    # Customize category order for sexual orientation
    if IDENTITY == "sexual_orientation":
        category_order = [
            "Straight or heterosexual",
            "Gay or Lesbian",
            "Bisexual and/or Pansexual",
            "Queer",
            "Asexual or aromantic",
            "Sexually fluid",
            "Demisexual",
            "Questioning",
            "Multiple Identities",
            "Other",
            "I prefer not to answer"
        ]
        category_order = [cat for cat in category_order if cat in proportions.index]
        proportions = proportions.loc[category_order]

    # Get colors
    color_mapping = _get_color_mapping(IDENTITY, custom_colors)
    if color_mapping:
        colors = [color_mapping.get(category, "#ACACAC") for category in proportions.index]
    else:
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(proportions.index)))

    plot_df = proportions * 100
    plot_df.T.plot(kind='barh', stacked=True, ax=ax, color=colors)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))

    ax.set_ylabel("")
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add R² values if provided
    _add_r2_values(ax, proportions, r2_dict, column_mapping)
    
    # Configure legend with custom order and labels
    legend_labels = [
        ("Straight or heterosexual", "Straight"),
        ("Gay or Lesbian", "Gay/Lesbian"),
        ("Bisexual and/or Pansexual", "Bisexual/\nPansexual"),
        ("Queer", "Queer"),
        ("Asexual or aromantic", "Asexual/\nAromantic"),
        ("Sexually fluid", "Sexually Fluid"),
        ("Demisexual", "Demisexual"),
        ("Questioning", "Questioning"),
        ("Multiple Identities", "Multiple"),
        ("Other", "Other"),
        ("I prefer not to answer", "No Answer")
    ]
    legend_labels = [(cat, label) for cat, label in legend_labels if cat in proportions.index]
    handles = [patches.Patch(color=color_mapping.get(cat, "#ACACAC"), label=label) for cat, label in legend_labels]
    legend = ax.legend(
        handles=handles,
        title=f'{" ".join(IDENTITY.split("_")) .title()}',
        bbox_to_anchor=has_custom_bbox,
        loc='best',
        frameon=True,
        framealpha=0.9,
        facecolor='white',
        edgecolor='lightgray'
    )
    legend.get_title().set_fontweight('bold')
    
    # Add alternating row backgrounds
    for i in range(len(proportions.columns)):
        if i % 2 == 0:
            ax.axhspan(i-0.4, i+0.4, color='#f5f5f5', zorder=0)
    
    # Clean up plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if no_legend:
        plt.legend('', frameon=False)
    plt.tight_layout()
    plt.show()
    
    return fig