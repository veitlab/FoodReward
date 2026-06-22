"""
Code for results depicted in Supplementary Figure 1
    A) Click accuracy during training

Functions for the code repository accompanying:
Birdsong Modification with Food Reward

Author: Franziska Heubach
Year: 2026
"""

# ----- Imports ----- #

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from util.helper_fct import (
    DATA_DIR,
    get_figure_dir
)

from util.helper_fct import (
    sem_of_valid, load_click_accuracy_percentages,
    make_click_accuracy_summary, print_click_accuracy_summary
)

bird_colors = {
    "Bird1_bkd": "orange",
    "Bird2_J6": "#6b96ffff",
    "Bird2_B3": "#26d9e4ff",
    "Bird2_DC4": "#1b35cfff",
    "Bird3_bF5": "red",
    "Bird3_kB4": "#942d2dff",
    "Bird4_HF4": "#1F7426",
    "Bird4_J5": "#7BE966",
    "Bird5_A11": "pink"
}

#----- Load data -----#

file_path = DATA_DIR / "click_accuracy.xlsx"

save_dir = get_figure_dir("FigureS1")

#----- Global plot adjustments -----#

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False


# ----- Plot functions ----- #

def plot_click_accuracy_summary(
    summary_df,
    sheet_colors,
    figsize=(1.5, 1.591),
    save_path=None,
    dpi=300,
):
    """
    Plot mean accuracy for categories, here "hits", "misses", "false hits"

    Parameters
    ----------
    summary_df : df
        Contains data for each category for each experiment
    sheet_colors : dict
        Custom color map for all experiments
    """
    x_pos = np.arange(len(CATEGORIES))
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=figsize)

    # Individual sheet values
    for i, category in enumerate(CATEGORIES):
        cat_df = summary_df[summary_df["category"] == category]

        jitter = rng.normal(
            loc=0,
            scale=0.04,
            size=len(cat_df),
        )

        xs = x_pos[i] + jitter

        for x, (_, row) in zip(xs, cat_df.iterrows()):
            ax.scatter(
                x,
                row["value"],
                s=10,
                color=sheet_colors[row["sheet"]],
                alpha=0.85,
                zorder=2,
            )

    # Mean ± SEM
    for i, category in enumerate(CATEGORIES):
        vals = summary_df.loc[
            summary_df["category"] == category,
            "value"
        ].astype(float)

        mean_val = vals.mean()
        sem_val = sem_of_valid(vals)

        ax.errorbar(
            x_pos[i],
            mean_val,
            yerr=sem_val,
            fmt="o",
            color="black",
            lw=1.2,
            capsize=3,
            markersize=4,
            zorder=4,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(CATEGORY_LABELS)
    ax.set_ylabel("Rate [%]")
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            label=sheet,
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=4,
        )
        for sheet, color in sheet_colors.items()
    ]

    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
        )
        print(f"Figure saved to: {save_path}")

    plt.show()

# ----- Run Main----- #

if __name__ == "__main__":

    # Fig S1 A
    CATEGORIES = ["hits", "misses", "false hits"]
    CATEGORY_LABELS = ["H", "M", "FH"]

    all_pct = load_click_accuracy_percentages(
        file_path=file_path,
        categories=CATEGORIES
        )
    
    summary_df = make_click_accuracy_summary(all_pct, CATEGORIES)

    print_click_accuracy_summary(summary_df, CATEGORIES)

    save_path = save_dir / "click_accuracy.svg"

    plot_click_accuracy_summary(
        summary_df=summary_df,
        sheet_colors=bird_colors,
        figsize=(1.5, 1.591),
        save_path=save_dir / "FigS1_A.svg"
        )
