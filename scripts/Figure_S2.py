"""
Code for results depicted in Supplementary Figure 2
    A + B + C) Repeat distributions for baseline and training

Functions for the code repository accompanying:
Birdsong Modification with Food Reward

Author: Franziska Heubach
Year:2026
"""

#----- Imports -----#

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

from util.helper_fct import (
    load_birds,
    get_figure_dir
)

from util.helper_fct import (
    sem_of_valid, normality_from_values,
    get_repeat_lengths, day_sort_key
)

#----- Load data -----#

birds = load_birds()
birds_catch = load_birds(catch=True)

save_dir = get_figure_dir("FigureS2")

#----- Global plot adjustments -----#

plt.rcParams['svg.fonttype'] = 'none' 
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

#----- Plot functions -----#

def plot_repeat_distributions_per_phase(
    labels_dict,
    target_syl,
    ylim = None,
    context=False,
    n_last_train_days=3,
    color="lightgreen",
    figsize=(6, 5),
    save_dir=None,
    save_name=None,
    dpi=300
):
    """
    Plot repeat number distributions for baseline and training data,
    for n last days of training.
    
    Parameters
    ----------
    labels_dict : dict
        Dictionary containing label sequences over one or multiple days
    target_syl : str
        Syllable for which the repeat number should be calculated
    y-lim : int 
        Will set a y-axis limit
        default : None
    context : bool
        if True, count context-dependent repeats, 
        thus only counts repeating syllable in the preceding context
        (e.g. "kb" only counts b+ with k preceding)
    n_last_train_days : int
        Number of last training days to consider,
        default : 3
        
    Returns
    -------
        day_data : dict
            Dictionary with repeat "freq", "counts", and "lengths" for each day
    """

    # collect repeat lengths per day
    per_day_lengths = defaultdict(list)

    for file_path, labels in labels_dict.items():
        if not labels:
            continue

        day_folder = os.path.basename(os.path.dirname(file_path))
        day_low = day_folder.lower()

        # only baseline and training
        if not (day_low.startswith("base") or day_low.startswith("train")):
            continue

        lengths = get_repeat_lengths(
            labels,
            target_syl,
            local_context=context
        )

        lengths = np.asarray(lengths, dtype=float)
        lengths = lengths[np.isfinite(lengths)]

        if lengths.size > 0:
            per_day_lengths[day_folder].extend(lengths.astype(int).tolist())

    days = sorted(per_day_lengths.keys(), key=day_sort_key)
    
    if len(days) == 0:
        print("No baseline or training days with matches found.")
        return None

    train_days = [
        d for d in days
        if d.lower().startswith("train")
    ]

    last_train_days = train_days[-n_last_train_days:]

    plot_days = [
        d for d in days
        if d.lower().startswith("base") or d in last_train_days
    ]

    all_lengths_for_axis = [
        length
        for vals in per_day_lengths.values()
        for length in vals
    ]

    if len(all_lengths_for_axis) == 0:
        print("No repeat lengths found.")
        return None

    max_len = max(all_lengths_for_axis)
    xs = np.arange(max_len + 1)

    baseline_counts = []
    training_counts = []

    baseline_values = []
    training_values = []

    day_data = {}

    for day in plot_days:
        lengths = np.asarray(per_day_lengths[day], dtype=int)

        counts = np.bincount(lengths, minlength=len(xs))
        total = counts.sum()
        freq = counts / total if total > 0 else np.zeros_like(xs, dtype=float)

        day_data[day] = {
            "freq": freq,
            "counts": counts,
            "lengths": lengths
        }

        day_low = day.lower()

        if day_low.startswith("base"):
            baseline_counts.append(counts)
            baseline_values.extend(lengths)

        elif day_low.startswith("train"):
            training_counts.append(counts)
            training_values.extend(lengths)

    baseline_values = np.asarray(baseline_values, dtype=float)
    training_values = np.asarray(training_values, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    if len(baseline_counts) > 0:
        baseline_total_counts = np.vstack(baseline_counts).sum(axis=0)
        baseline_total = baseline_total_counts.sum()
        baseline_freq = baseline_total_counts / baseline_total if baseline_total > 0 else np.zeros_like(xs, dtype=float)

        ax.plot(
            xs,
            baseline_freq,
            linewidth=1,
            color="black",
            marker="o",
            markersize=1.5,
            label="B"
        )

    if len(training_counts) > 0:
        training_total_counts = np.vstack(training_counts).sum(axis=0)
        training_total = training_total_counts.sum()
        training_freq = training_total_counts / training_total if training_total > 0 else np.zeros_like(xs, dtype=float)

        ax.plot(
            xs,
            training_freq,
            linewidth=1,
            color=color,
            linestyle="-",
            marker="o",
            markersize=1.5,
            label="T"
        )

    # --- mean markers ---
    if baseline_values.size > 0:
        mean_baseline = float(np.mean(baseline_values))
        ax.axvline(
            mean_baseline,
            color="black",
            linestyle="--",
            linewidth=1
        )
        ax.text(
            mean_baseline + 0.05,
            0.96,
            f"{mean_baseline:.1f}",
            color="black",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top"
        )

    if training_values.size > 0:
        mean_training = float(np.mean(training_values))
        ax.axvline(
            mean_training,
            color=color,
            linestyle="--",
            linewidth=1
        )
        ax.text(
            mean_training + 0.05,
            0.90,
            f"{mean_training:.1f}",
            color=color,
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top"
        )

    print("\n=== Repeat distribution summary ===")

    if baseline_values.size > 0:
        baseline_sem = sem_of_valid(baseline_values)
        print(
            f"Baseline: n={baseline_values.size}, "
            f"mean={np.mean(baseline_values):.4f}, "
            f"sem={baseline_sem:.4f}"
        )
    else:
        print("[warn] No baseline values found.")

    if training_values.size > 0:
        training_sem = sem_of_valid(training_values)
        print(
            f"Training catch: n={training_values.size}, "
            f"mean={np.mean(training_values):.4f}, "
            f"sem={training_sem:.4f}"
        )
    else:
        print("[warn] No training catch values found.")

    # --- statistical test: baseline vs training ---
    print("\n=== Test: Baseline vs Training ===")

    if baseline_values.size == 0 or training_values.size == 0:
        print("[warn] Test not possible because one group is empty.")

    else:
        baseline_norm = normality_from_values(baseline_values)
        training_norm = normality_from_values(training_values)

        can_ttest = (
            baseline_norm["normal"]
            and training_norm["normal"]
            and baseline_values.size >= 2
            and training_values.size >= 2
        )

        if can_ttest:
            test_name = "Welch t-test"
            test_out = stats.ttest_ind(
                baseline_values,
                training_values,
                equal_var=False
            )
        else:
            test_name = "Mann-Whitney-U"
            test_out = stats.mannwhitneyu(
                baseline_values,
                training_values,
                alternative="two-sided"
            )

        stat_val = float(test_out.statistic)
        p_val = float(test_out.pvalue)

        sig_txt = "significant" if p_val < 0.05 else "not significant"

        print(
            f"Comparison Baseline vs Training catch last {n_last_train_days} days: "
            f"Test={test_name}, "
            f"Stat={stat_val:.4f}, "
            f"p={p_val:.4g} -> {sig_txt}"
        )

    ax.set_xlabel(f"Repeat Number {target_syl}")
    ax.set_ylabel("Relative Frequency")

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top = ylim)

    x_max = int(np.max(xs)) if len(xs) > 0 else 0
    ax.set_xticks(np.arange(0, x_max + 1, 2))

    y_max = ax.get_ylim()[1]
    ax.set_yticks(np.arange(0, y_max + 0.001, 0.1))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper right",
        ncol=1,
        frameon=False
    )

    fig.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        if save_name is None:
            safe_target = re.sub(r"[^A-Za-z0-9]+", "_", target_syl).strip("_")
            save_name = f"repeat_distribution_{safe_target}.png"

        save_path = os.path.join(save_dir, save_name)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()

    return day_data

#----- Run Main -----#

if __name__ == "__main__":

    # Fig S2 A "B3"
    plot_repeat_distributions_per_phase(
        labels_dict=birds_catch["Bird2_B3"],
        target_syl="b+",
        color="#26d9e4ff",
        figsize=(1.9, 1.5),
        save_dir=get_figure_dir("FigureS2"),
        save_name="FigS2_A.svg",
        dpi=300
    )

    # Fig S2 B "bF"
    plot_repeat_distributions_per_phase(
        labels_dict=birds_catch["Bird3_bF5"],
        target_syl="bf+",
        context=True,
        color="red",
        figsize=(1.9, 1.5),
        save_dir=get_figure_dir("FigureS2"),
        save_name="FigS2_B.svg",
        dpi=300
    )

    # Fig S2 C "J"
    # I am using 'jm' here as the same syllable can be labelled 'j' or 'm' in the original sequence
    plot_repeat_distributions_per_phase(
        labels_dict=birds_catch["Bird4_J5"],
        target_syl="[jm]+",
        color="#7BE966",
        figsize=(1.9, 1.5),
        save_dir=get_figure_dir("FigureS2"),
        save_name="FigS2_C.svg",
        dpi=300
    )