"""
Code for results depicted in Supplementary Figure 3
    A) Relative learning trajectory branch point over training days
       Relative learning trajectory repeats over training days
    C) Overview target repeat phrase occurrence during baseline and training


Functions for the code repository accompanying:
Birdsong Modification with Food Reward

Author: Franziska Heubach
Year: 2026
"""

# ----- Imports ----- #

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import MultipleLocator
from scipy.stats import wilcoxon

from util.helper_fct import (
    load_birds,
    get_figure_dir,
    BIRD_COLORS as bird_colors,
    BIRD_NAME_MAPPING as bird_name_mapping,
    TARGET_SYL_PER_BIRD as target_syl_per_bird,
    TARGET_REPEAT_PER_BIRD as target_repeat_per_bird,
    CONTEXT_PER_BIRD as context_per_bird
)

from util.helper_fct import (
    compute_branch_probs, compute_day_raw_values,
    split_baseline_training_days, sem_of_valid,
    stars_from_p, get_repeat_lengths, map_bird_name)

# ----- Load data ----- # 

birds_catch_alldays = load_birds(source="alldays", catch=True)
birds_alldays = load_birds(source="alldays")

save_dir = get_figure_dir("FigureS3")

# ----- Global plot adjustments ----- #

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False


# ----- Plot functions ----- #

def plot_branchpoint_learning_trajectory(
    bird_labels_dict,
    target_syl_per_bird,
    branch_syl_per_bird,
    target_branch_per_bird,
    bird_colors=None,
    figsize=(5, 6),
    save_dir=None,
    save_name=None,
    dpi=300,
):
    """
    Plot relative branchpoint learning across training days.

    Normalize each bird's transition probabilities to its mean
    baseline probability (0%) and the mean probability on calendar
    days T4–T6 (100%).

    Training days are counted from the last baseline day.

    Parameters
    ----------
    bird_labels_dict : dict
        Mapping of bird IDs to dictionaries of file paths and sequences.
    target_syl_per_bird : dict
        Branchpoint syllable for each bird.
    branch_syl_per_bird : dict
        List of possible following syllables for each bird.
    target_branch_per_bird : dict
        Target following syllable for each bird.
    bird_colors : dict, optional
        Mapping of bird IDs to plot colors.
    """
    bird_colors = bird_colors or {}

    fig, ax = plt.subplots(figsize=figsize)
    individual_data = {}

    for bird, labels in bird_labels_dict.items():
        target_syl = target_syl_per_bird[bird]
        branch_syl = branch_syl_per_bird[bird]
        target_branch = target_branch_per_bird[bird]

        if target_branch not in branch_syl:
            raise ValueError(
                f"{bird}: target branch {target_branch!r} not in {branch_syl!r}"
            )

        matrix, _, folders = compute_branch_probs(
            labels,
            target_syl,
            branch_syl,
            show_postbase=False,
        )
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != len(folders):
            raise ValueError(
                f"{bird}: unexpected matrix shape {matrix.shape}"
            )
        if matrix.shape[1] < len(branch_syl):
            raise ValueError(
                f"{bird}: not enough branch columns: {matrix.shape}"
            )
        idx = branch_syl.index(target_branch)
        day_prob = dict(zip(folders, matrix[:, idx]))
        base_days = [
            day for day in folders
            if day.lower().startswith("base")
        ]
        train_days = [
            day for day in folders
            if day.lower().startswith("train")
        ]
        if not base_days or not train_days:
            print(f"[skip] {bird}: missing baseline or training")
            continue
        # Extract and validate date suffixes
        all_days = base_days + train_days
        if any(re.search(r"\d{6}$", day) is None for day in all_days):
            raise ValueError(
                f"{bird}: missing YYMMDD date suffix in folder names"
            )
        base_days = sorted(
            base_days,
            key=lambda day: datetime.strptime(day[-6:], "%y%m%d"),
        )
        train_days = sorted(
            train_days,
            key=lambda day: datetime.strptime(day[-6:], "%y%m%d"),
        )
        base = np.array(
            [day_prob[day] for day in base_days],
            dtype=float,
        )
        base = base[np.isfinite(base)]
        if base.size == 0:
            print(f"[skip] {bird}: no finite baseline probability")
            continue
        baseline = float(np.mean(base))
        y_abs = np.array(
            [day_prob[day] for day in train_days],
            dtype=float,
        )
        last_baseline_date = datetime.strptime(
            base_days[-1][-6:], "%y%m%d"
        )
        x = np.array([
            (
                datetime.strptime(day[-6:], "%y%m%d")
                - last_baseline_date
            ).days
            for day in train_days
        ])
        # Mean probability across T4–T6 as  100% reference
        mask = np.isin(x, [4, 5, 6]) & np.isfinite(y_abs)
        if not np.any(mask):
            print(f"[skip] {bird}: no valid T4–T6")
            continue

        final = float(np.mean(y_abs[mask]))
        delta = final - baseline
        if not np.isfinite(delta) or np.isclose(delta, 0):
            print(f"[skip] {bird}: final probability equals baseline")
            continue
        y_rel = 100 * (y_abs - baseline) / delta
        y_full = np.full(7, np.nan)
        y_full[0] = 0
        for day, learning in zip(x, y_rel):
            if 1 <= day <= 6 and np.isfinite(learning):
                y_full[day] = learning
        individual_data[bird] = y_full

        print(
            f"\n{bird}: baseline={baseline:.4f}, "
            f"final={final:.4f}, delta={delta:.4f}"
        )

        for folder, day, prob, learning in zip(
            train_days, x, y_abs, y_rel
        ):
            print(
                f"  {folder}: T{day}, "
                f"p={prob:.4f}, learning={learning:.1f}%"
            )

    for i, (bird, y_values) in enumerate(individual_data.items()):
        color = bird_colors.get(bird, f"C{i}")
        ax.plot(
            np.arange(7),
            y_values,
            marker="o",
            markersize=3,
            linewidth=1,
            color=color,
            alpha=1,
            zorder=2,
        )

    ax.axhline(100, color="black", linestyle="--")
    ax.axhline(0, color="black")
    ax.set_xlabel("Training Days")
    ax.set_ylabel("Percent of\nFinal Learning")
    ax.set_xlim(-0.2, 6.2)
    ax.set_xticks(np.arange(7))
    ax.yaxis.set_major_locator(MultipleLocator(25))
    fig.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(
            save_dir,
            save_name or "branchpoint_final_learning.svg",
        )
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {path}")

    return fig, ax

def plot_repeat_learning_trajectory(
    bird_labels_dict,
    target_syl_per_bird,
    context_per_bird,
    figsize=(5, 6),
    save_dir=None,
    save_name=None,
    dpi=300,
):
    """
    Plot relative target-repeat learning across training days.

    Normalize each bird's daily mean repeat number to its mean
    baseline value (0%) and the pooled mean of all repeat
    occurrences across T4-T6 (100%).

    Display group mean ± SEM and test T1-T5 against 100%.
    Training days are counted from the last baseline day.

    Parameters
    ----------
    bird_labels_dict : dict
        Mapping of bird IDs to dictionaries of file paths and sequences.
    target_syl_per_bird : dict
        Target-repeat pattern for each bird.
    context_per_bird : dict
        Whether repeat counting is context-dependent for each bird.
    """
    average_data = {0: []}

    for bird, labels_dict in bird_labels_dict.items():
        day_data = compute_day_raw_values(
            labels_dict,
            target_syl=target_syl_per_bird[bird],
            context=context_per_bird[bird],
        )
        baseline_days, training_days = split_baseline_training_days(day_data)
        baseline_days = sorted(
            baseline_days,
            key=lambda item: datetime.strptime(item[0][-6:], "%y%m%d"),
        )
        training_days = sorted(
            training_days,
            key=lambda item: datetime.strptime(item[0][-6:], "%y%m%d"),
        )
        if not baseline_days or not training_days:
            print(f"[skip] {bird}: missing baseline or training")
            continue
        baseline_values = [
            np.mean(values)
            for _, values in baseline_days
            if len(values) > 0
        ]
        if not baseline_values:
            print(f"[skip] {bird}: no valid baseline values")
            continue
        baseline = float(np.mean(baseline_values))
        last_baseline = datetime.strptime(
            baseline_days[-1][0][-6:], "%y%m%d"
        )
        x = np.array([
            (datetime.strptime(day[-6:], "%y%m%d") - last_baseline).days
            for day, _ in training_days
        ])
        y_abs = np.array([
            np.mean(values) if len(values) > 0 else np.nan
            for _, values in training_days
        ])
        # Pool individual repeat occurrences across T4-T6
        final_lengths = [
            length
            for (_, values), day in zip(training_days, x)
            if day in (4, 5, 6)
            for length in values
            if np.isfinite(length)
        ]
        if not final_lengths:
            print(f"[skip] {bird}: no valid T4-T6 values")
            continue
        final = float(np.mean(final_lengths))
        delta = final - baseline
        if not np.isfinite(delta) or np.isclose(delta, 0):
            print(f"[skip] {bird}: final mean equals baseline")
            continue
        y_rel = 100 * (y_abs - baseline) / delta
        average_data[0].append(0.0)
        for day, value in zip(x, y_rel):
            if 1 <= day <= 6 and np.isfinite(value):
                average_data.setdefault(int(day), []).append(float(value))
        print(
            f"\n{bird}: baseline={baseline:.3f}, "
            f"final={final:.3f}, delta={delta:.3f}"
        )
        for (folder, _), day, raw, rel in zip(
            training_days, x, y_abs, y_rel
        ):
            print(
                f"  {folder}: T{day}, "
                f"raw={raw:.3f}, learning={rel:.1f}%"
            )

    if not average_data[0]:
        print("No valid birds available for plotting.")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    days = sorted(average_data)
    means = [
        float(np.mean(average_data[day]))
        for day in days
    ]
    sems = [
        sem_of_valid(average_data[day])
        for day in days
    ]

    for day, mean, sem in zip(days, means, sems):
        print(
            f"T{day}: Mean={mean:.2f}, SEM={sem:.2f}, "
            f"n={len(average_data[day])}"
        )

    ax.errorbar(
        days,
        means,
        yerr=sems,
        fmt="o-",
        color="black",
        linewidth=1,
        markersize=3,
        capsize=3,
        capthick=1,
        elinewidth=1,
        zorder=10,
    )

    # Wilcoxon tests against final learning
    all_values = [
        value
        for values in average_data.values()
        for value in values
    ]
    y_range = max(all_values) - min(all_values)
    star_y = max(all_values) + 0.08 * max(y_range, 1)

    for day in range(1, 6):
        values = np.asarray(average_data.get(day, []), dtype=float)

        if values.size < 2:
            print(f"T{day}: n={values.size}, not enough data")
            continue
        differences = values - 100
        if np.allclose(differences, 0):
            statistic, p = 0.0, 1.0
        else:
            result = wilcoxon(differences)
            statistic, p = result.statistic, result.pvalue
        print(
            f"T{day}: n={values.size}, "
            f"W={statistic:.3f}, p={p:.4f}"
        )
        if p < 0.05:
            ax.text(
                day,
                star_y,
                stars_from_p(p),
                ha="center",
                va="bottom",
            )

    ax.axhline(100, linestyle="--", linewidth=1, color="black")
    ax.axhline(0, linewidth=1, color="black")
    ax.set_xlabel("Training Days")
    ax.set_ylabel("Percent of final learning")
    ax.set_xlim(-0.2, 6.2)
    ax.set_xticks(np.arange(7))
    ax.yaxis.set_major_locator(MultipleLocator(25))
    fig.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            save_name or "target_repeat_final_learning.svg",
        )
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig, ax

def plot_target_repeat_renditions(
    bird_labels_dict,
    target_syl_per_bird,
    target_repeat_per_bird,
    context_per_bird,
    bird_colors=None,
    bird_name_mapping=None,
    figsize=(13, 7),
    point_size=6,
    point_alpha=0.5,
    shade_alpha=0.15,
    save_dir=None,
    save_name=None,
    dpi=300,
    ):
    """
    Plot individual target-repeat occurrences during baseline and training.

    Show baseline occurrences in the left panel and consecutive
    occurrences from training days T1-T6 in the right panel.
    Each row represents one experiment and each point one repeat
    occurrence. Shading indicates the target repeat criterion.

    Training days are counted from the last baseline day.

    Parameters
    ----------
    bird_labels_dict : dict
        Mapping of bird IDs to dictionaries of file paths and sequences.
    target_syl_per_bird : dict
        Target-repeat pattern for each bird.
    target_repeat_per_bird : dict
        Required repeat number for each bird.
    context_per_bird : dict
        Whether repeat counting is context-dependent for each bird.
    bird_colors : dict, optional
        Mapping of bird IDs to plot colors.
    bird_name_mapping : dict, optional
        Mapping of bird prefixes to display names.
    point_size : float
        Scatter marker area in points squared.
    point_alpha : float
        Opacity of the occurrence markers.
    shade_alpha : float
        Opacity of the target criterion shading.
    """
    bird_colors = bird_colors or {}
    bird_name_mapping = bird_name_mapping or {}
    plot_data = {}

    for bird, labels_dict in bird_labels_dict.items():
        target_syl = target_syl_per_bird[bird]
        target_repeat = target_repeat_per_bird[bird]
        context = context_per_bird[bird]

        day_files = {}

        for file_path, sequence in labels_dict.items():
            day = os.path.basename(os.path.dirname(file_path))

            if day.lower().startswith(("base", "train")):
                day_files.setdefault(day, []).append((file_path, sequence))

        base_days = sorted(
            (day for day in day_files if day.lower().startswith("base")),
            key=lambda day: datetime.strptime(day[-6:], "%y%m%d"),
        )
        train_days = sorted(
            (day for day in day_files if day.lower().startswith("train")),
            key=lambda day: datetime.strptime(day[-6:], "%y%m%d"),
        )

        if not base_days or not train_days:
            print(f"[skip] {bird}: missing baseline or training")
            continue

        last_baseline = datetime.strptime(base_days[-1][-6:], "%y%m%d")

        baseline_lengths = []

        for day in base_days:
            for _, sequence in sorted(day_files[day]):
                baseline_lengths.extend(
                    get_repeat_lengths(sequence, target_syl, context)
                )

        training_data = {}

        for day in train_days:
            training_day = (
                datetime.strptime(day[-6:], "%y%m%d") - last_baseline
            ).days

            if not 1 <= training_day <= 6:
                continue

            lengths = []

            for _, sequence in sorted(day_files[day]):
                lengths.extend(
                    get_repeat_lengths(sequence, target_syl, context)
                )

            training_data[training_day] = lengths

            n_target = sum(length >= target_repeat for length in lengths)
            print(
                f"{bird} | T{training_day}: "
                f"{n_target}/{len(lengths)} renditions >= target"
            )

        plot_data[bird] = {
            "baseline": baseline_lengths,
            "training": training_data,
        }

    if not plot_data:
        raise ValueError("No valid bird data found.")

    fig, axes = plt.subplots(
        len(plot_data),
        2,
        figsize=figsize,
        sharex=False,
        sharey=False,
        squeeze=False,
        gridspec_kw={"width_ratios": [1, 4]},
    )

    for row, (bird, data) in enumerate(plot_data.items()):
        ax_base, ax_train = axes[row]

        color = bird_colors.get(bird, "black")
        display_name = map_bird_name(bird, bird_name_mapping)
        target_repeat = target_repeat_per_bird[bird]

        baseline_lengths = data["baseline"]
        training_data = data["training"]

        all_lengths = list(baseline_lengths)

        for day in range(1, 7):
            all_lengths.extend(training_data.get(day, []))

        bird_ymax = (
            max(max(all_lengths), target_repeat) + 0.5
            if all_lengths
            else target_repeat + 0.5
        )

        # Baseline occurrences
        if baseline_lengths:
            x_base = np.arange(1, len(baseline_lengths) + 1)

            ax_base.scatter(
                x_base,
                baseline_lengths,
                s=point_size,
                color=color,
                alpha=point_alpha,
                edgecolors="none",
                zorder=2,
            )
            ax_base.set_xlim(0, len(baseline_lengths) + 1)

        ax_base.set_ylabel(display_name)

        # Consecutive training occurrences
        current_x = 0
        day_boundaries = []

        for day in range(1, 7):
            lengths = training_data.get(day, [])

            if not lengths:
                continue

            x = np.arange(current_x + 1, current_x + len(lengths) + 1)

            ax_train.scatter(
                x,
                lengths,
                s=point_size,
                color=color,
                alpha=point_alpha,
                edgecolors="none",
                zorder=2,
            )

            current_x = int(x[-1])
            day_boundaries.append(current_x + 0.5)

        for boundary in day_boundaries[:-1]:
            ax_train.axvline(
                boundary,
                color="black",
                linewidth=0.8,
                alpha=0.6,
                zorder=1,
            )

        # Target region and shared y-range within each row
        for ax in (ax_base, ax_train):
            ax.axhspan(
                target_repeat - 0.5,
                bird_ymax,
                color=color,
                alpha=shade_alpha,
                zorder=0,
            )
            ax.set_ylim(0.5, bird_ymax)
            ax.set_yticks(
                np.arange(1, int(np.floor(bird_ymax)) + 1)
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax_base.tick_params(axis="both", length=3)

        ax_train.tick_params(axis="y", left=False, labelleft=False)
        ax_train.tick_params(axis="x", length=3)
        ax_train.spines["left"].set_visible(False)

        if current_x > 0:
            ax_train.set_xlim(0, current_x + 1)
            ax_train.set_xticks(np.arange(0, current_x + 1, 500))

        if row == 0:
            ax_base.text(
                0.5, 1.02, "Baseline",
                transform=ax_base.transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax_train.text(
                0.5, 1.02, "Training Days",
                transform=ax_train.transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[-1, 0].set_xlabel("Baseline Occurrence")
    axes[-1, 1].set_xlabel("Target Repeat Phrase Occurrence")

    fig.subplots_adjust(
        left=0.11,
        right=0.995,
        top=0.97,
        bottom=0.07,
        hspace=0.35,
        wspace=0.05,
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            save_name or "target_repeat_renditions_long.svg",
        )
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig, axes, plot_data

# ----- Run Main ----- #

if __name__ == "__main__":

    # Fig S3 A branch points
    plot_branchpoint_learning_trajectory(
        bird_labels_dict={
            bird: data
            for bird, data in birds_catch_alldays.items()
            if bird in ("Bird1_bkd", "Bird6_ah")
        },
        target_syl_per_bird={"Bird1_bkd": "bb", "Bird6_ah": "a"},
        branch_syl_per_bird={"Bird1_bkd": ["d", "i"], "Bird6_ah": ["h", "t"]},
        target_branch_per_bird={"Bird1_bkd": "d", "Bird6_ah": "h"},
        bird_colors=bird_colors,
        figsize=(1.7, 1.3),
        save_dir=save_dir,
        save_name="S3_Fig_A_branchpoint.svg",
        dpi=300,
    )

    # Fig S3 A repeats
    plot_repeat_learning_trajectory(
        bird_labels_dict={
            bird: data
            for bird, data in birds_catch_alldays.items()
            if bird not in ("Bird1_bkd", "Bird6_ah")
        },
        target_syl_per_bird=target_syl_per_bird,
        context_per_bird=context_per_bird,
        figsize=(1.7, 1.3),
        save_dir=save_dir,
        save_name="S3_Fig_A_repeat.svg",
        dpi=300,
    )

    # Fig S3 C 
    plot_target_repeat_renditions(
        bird_labels_dict={
            bird: data
            for bird, data in birds_alldays.items()
            if bird not in ("Bird1_bkd", "Bird6_ah")
        },
        target_syl_per_bird=target_syl_per_bird,
        target_repeat_per_bird=target_repeat_per_bird,
        context_per_bird=context_per_bird,
        bird_name_mapping=bird_name_mapping,
        bird_colors=bird_colors,
        figsize=(7, 4),
        point_size=4,
        save_dir=save_dir,
        save_name="S3_Fig_C.svg",
        dpi=300,
    )