"""
Code for results depicted in Supplementary Figure 1
    A) Approach probability during conditioning phase
    B) Approach latency during conditioning phase
    C) Approach latency during training
    D) Click accuracy during training

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
import pandas as pd

from util.helper_fct import (
    DATA_DIR,
    get_figure_dir,
    BIRDS,
    CONTROL_BIRDS,
    BIRD_COLORS as bird_colors,
    BIRD_NAME_MAPPING as bird_name_mapping
)

from util.helper_fct import (
    sem_of_valid, load_click_accuracy_percentages, map_bird_name,
    make_click_accuracy_summary, print_click_accuracy_summary
)

bird_colors = {
    **bird_colors,
    "Bird5_A11":"pink"
}

#----- Load data -----#

file_path_acc = DATA_DIR / "Click_Accuracy.xlsx"
file_path_cond = DATA_DIR / "Conditioning.xlsx"
file_path_train = DATA_DIR / "Approach_Training.xlsx"

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

def plot_approach_probability(
    excel_path,
    sheet_name,
    bird_colors=None,
    jitter=0.035,
    figsize=(3, 4),
    save_path=None,
    dpi=300,
    skiprows=2
    ):
    """
    Plot approach probability on the final conditioning day.

    Calculate the mean approach probability for each bird across valid
    trials and display individual birds and group mean ± SEM.

    Parameters
    ----------
    excel_path : str or Path
        Path to the conditioning Excel file.
    sheet_name : str
        Name of the worksheet containing approach data.
    bird_colors : dict
        Mapping of bird IDs to plotting colors.
    jitter : float
        Horizontal jitter applied to individual data points.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=skiprows)

    bird_cols = [col for col in df.columns if col != "Trial"]
    approach_prob = df[bird_cols].apply(pd.to_numeric, errors="coerce").mean()
    approach_prob = approach_prob.dropna()
    mean = approach_prob.mean()
    sem = approach_prob.sem()

    if bird_colors is None:
        bird_colors = {}

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        1, mean, yerr=sem,
        fmt="o",
        color="black",
        markersize=4,
        capsize=3,
        linewidth=1,
        zorder=5
    )
    rng = np.random.default_rng(42)
    x = 1 + rng.uniform(-jitter, jitter, len(approach_prob))
    for xx, (bird, probability) in zip(x, approach_prob.items()):
        ax.scatter(
            xx, probability,
            color=bird_colors.get(bird, "gray"),
            edgecolor=bird_colors.get(bird, "gray"),
            linewidth=0.8,
            s=12,
            alpha=0.9,
            zorder=4
        )

    ax.set_xticks([1])
    ax.set_xticklabels(["Final Day"])
    ax.set_xlim(0.75, 1.25)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Approach Probability")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    print(f"Mean approach probability: {mean:.3f} ± {sem:.3f} SEM")
    print(approach_prob)

    return fig, ax, approach_prob

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
    ax.set_xticklabels(CATEGORIES)
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

def plot_approach_latency(
    excel_path,
    sheet_name,
    bird_colors=None,
    bird_name_mapping=None,
    box_width=0.2,
    box_spacing=0.6,
    jitter=0.025,
    figsize=(3.5, 2.5),
    dpi=300,
    save_path=None
    ):
    """
    Plot approach latencies for individual birds.

    Plot latency measurements as colored scatter points with
    overlaid boxplots for each bird.

    Parameters
    ----------
    excel_path : str or Path
        Path to the Excel file containing latency data.
    sheet_name : str
        Worksheet containing latency data.
    bird_colors : dict, optional
        Mapping of bird IDs to plotting colors.
    bird_name_mapping : dict, optional
        Mapping of bird IDs to display names.
    box_width, box_spacing, jitter : float
        Parameters controlling plot layout and point positions.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=2)

    bird_cols = [col for col in df.columns if str(col).strip() != "Trial"]
    df[bird_cols] = df[bird_cols].apply(pd.to_numeric, errors="coerce")
    bird_colors = bird_colors or {}
    bird_name_mapping = bird_name_mapping or {}
    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(42)
    positions = np.arange(len(bird_cols)) * box_spacing
    for i, bird in enumerate(bird_cols):
        pos = positions[i]
        values = df[bird].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue

        color = bird_colors.get(bird, "gray")

        ax.boxplot(
            values,
            positions=[pos],
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops=dict(facecolor="none", edgecolor="black", linewidth=0.6),
            medianprops=dict(color="black", linewidth=0.6),
            whiskerprops=dict(color="black", linewidth=0.6),
            capprops=dict(color="black", linewidth=0.6),
            zorder=10
        )
        x = pos + rng.uniform(-jitter, jitter, len(values))
        ax.scatter(
            x, values,
            color=color,
            edgecolor=color,
            linewidth=0.3,
            s=5,
            alpha=0.7,
            zorder=5
        )
    display_names = [
        map_bird_name(bird, bird_name_mapping)
        for bird in bird_cols
    ]
    ax.set_xticks(positions)
    ax.set_xticklabels([name[4:5] for name in display_names])
    ax.set_xlim(positions[0] - 0.3, positions[-1] + 0.3)
    ax.set_xlabel("Bird")
    ax.set_ylabel("Approach Latency [s]")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax

def plot_approach_latency_over_days(
    excel_path,
    bird_colors=None,
    figsize=(1.7, 1.5),
    box_width=0.3,
    dpi=300,
    save_path=None,
    ):
    """
    Plot mean approach latency across training days.

    Calculate the mean latency per bird and day, display
    the distributions as boxplots and individual bird
    trajectories as colored lines.

    Parameters
    ----------
    excel_path : str or Path
        Path to the approach training Excel file.
    bird_colors : dict, optional
        Mapping of bird IDs to plotting colors.

    """
    bird_colors = bird_colors or {}

    sheets = pd.read_excel(excel_path, sheet_name=None, skiprows=4)

    latency_data = []

    for bird, df in sheets.items():
        df.columns = df.columns.astype(str).str.strip()
        required = ["Day", "Latency [s]"]
        if not all(col in df.columns for col in required):
            continue
        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        for day, group in df.groupby("Day"):
            latency_data.append({
                "bird": bird,
                "day": int(day),
                "latency": group["Latency [s]"].mean()
            })

    lat_df = pd.DataFrame(latency_data)
    lat_summary = (
        lat_df.groupby("day")["latency"]
        .agg(
            median="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
            min="min",
            max="max",
            n="count"
        )
        .reset_index()
    )

    print("\nApproach latency per day:")
    for _, row in lat_summary.iterrows():
        print(
            f"Day {int(row['day'])}: "
            f"Median = {row['median']:.2f} s, "
            f"IQR = {row['q1']:.2f}–{row['q3']:.2f} s, "
            f"Range = {row['min']:.2f}–{row['max']:.2f} s, "
            f"n = {int(row['n'])}"
        )

    fig, ax = plt.subplots(figsize=figsize)
    days = sorted(lat_df["day"].dropna().unique())
    positions = np.arange(len(days))

    for i, day in enumerate(days):
        values = lat_df.loc[
            lat_df["day"] == day, "latency"
        ].dropna().to_numpy()

        if len(values) == 0:
            continue

        ax.boxplot(
            values,
            positions=[positions[i]],
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops=dict(
                facecolor="none",
                edgecolor="black",
                linewidth=0.6
            ),
            medianprops=dict(color="black", linewidth=0.6),
            whiskerprops=dict(color="black", linewidth=0.6),
            capprops=dict(color="black", linewidth=0.6),
            zorder=10
        )

    for bird, group in lat_df.groupby("bird"):
        group = group.sort_values("day")
        color = bird_colors.get(bird, "gray")
        x = np.array([days.index(day) for day in group["day"]])
        y = group["latency"].to_numpy()
        ax.plot(
            x, y,
            color=color,
            linewidth=0.7,
            alpha=1,
            zorder=3
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(days)
    ax.set_xlabel("Day")
    ax.set_ylabel("Latency to\nApproach [s]")
    ax.set_ylim(0, 40)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=0.6, length=3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax, lat_df

# ----- Run Main----- #

if __name__ == "__main__":

    # Fig S1 A 
    fig, ax, approach_prob = plot_approach_probability(
        excel_path=file_path_cond,
        sheet_name="Approach_Sum",
        bird_colors=bird_colors,
        jitter=0.1,
        figsize=(1.0, 1.33),
        dpi=300,
        save_path=save_dir / "S1_Fig_A.svg",
    )

    # Fig S1 B 
    fig, ax = plot_approach_latency(
        excel_path=file_path_cond,
        sheet_name="Latency_Sum",
        bird_colors=bird_colors,
        bird_name_mapping=bird_name_mapping,
        jitter=0.06,
        figsize=(1.3, 1.5),
        dpi=600,
        save_path=save_dir / "S1_Fig_B.svg",
    )

    # Fig S1 C 
    fig, ax, lat_df = plot_approach_latency_over_days(
        excel_path=file_path_train,
        bird_colors=bird_colors,
        figsize=(1.7, 1.5),
        box_width=0.3,
        dpi=300,
        save_path=save_dir / "S1_Fig_C.svg",
    )

    # Fig S1 D Experiment birds
    CATEGORIES = ["hits", "misses", "false hits"]
    all_pct = load_click_accuracy_percentages(
        file_path=file_path_acc,
        categories=CATEGORIES
        )
    summary_df = make_click_accuracy_summary(all_pct, CATEGORIES)
    # Experiment birds
    experimental_birds = BIRDS + ["Bird5_A11"]
    experimental_df = summary_df[
        summary_df["sheet"].isin(experimental_birds)
    ].copy()
    experimental_colors = {
        bird: bird_colors[bird]
        for bird in experimental_birds
    }
    plot_click_accuracy_summary(
        summary_df=experimental_df,
        sheet_colors=experimental_colors,
        figsize=(1.5, 1.591),
        save_path=save_dir / "S1_Fig_D.svg",
        )
    # Control birds
    control_colors = {
        bird: bird_colors[bird]
        for bird in CONTROL_BIRDS
    }
    plot_click_accuracy_summary(
        summary_df=summary_df[
            summary_df["sheet"].isin(control_colors)
        ].copy(),
        sheet_colors=control_colors,
        figsize=(1.5, 1.591),
        save_path=save_dir / "S1_Fig_D_control.svg",
    )