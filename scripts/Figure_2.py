"""
Code for results depicted in Figure 2
    B + G) Repeat distributions for baseline and training
    C) Mean repeat number over training days
    D) Mean repeat number changes for all experiments
    E) Self-transition probability of the repeat syllables within the repeat phrase
    H) Self-transition probability gor meta repeat: within-chunk and between-chunk prob.

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
from matplotlib.lines import Line2D
from collections import defaultdict
from statsmodels.stats.multitest import multipletests

from util.helper_fct import (
    load_birds,
    get_figure_dir,
    BIRD_COLORS as bird_colors,
    BIRD_NAME_MAPPING as bird_name_mapping,
    TARGET_SYL_PER_BIRD as target_syl_per_bird,
    CONTEXT_PER_BIRD as context_per_bird,
    TARGET_REPEAT_PER_BIRD as target_repeat_per_bird
)

from util.helper_fct import (
    compute_day_raw_values,
    summarize_day_values,
    sem_of_valid, normality_from_values,
    plot_segments, get_repeat_lengths,
    day_sort_key, pool_all_lengths,
    nanmean_or_nan, map_bird_name,
    setup_bs_t_summary_axis,
    finalize_bs_t_summary_layout,
    stars_from_p, pooled_p_self,
    wilcoxon_by_training_from_summary
)

#----- Load data -----#

birds = load_birds()
birds_catch = load_birds(catch=True)

save_dir = get_figure_dir("Figure2")

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

def plot_mean_repeat_number_fullplot(
    labels_dict,
    target_syl,
    color_catch="lightblue",
    figsize=(9, 5),
    save_dir=None,
    save_name=None,
    dpi=300
):
    """
    Plots mean repeat number of target_syl per day/folder.
    Baseline, training, and postbaseline are all taken from the same dict.
    
    Parameters
    ----------
    labels_dict : dict
        Dictionary containing label sequences over one or multiple days
    target_syl : str
        Syllable for which the repeat number should be calculated
        
    Returns
    -------
    catch_mean : np.ndarray
        mean repeat number of target_syl for each day/folder
    all_folders : list [str]
        ordered list of all day folders
    """

    if labels_dict is None:
        print("[warn] labels_dict ist None.")
        return None

    catch_day_raw_values = compute_day_raw_values(
        labels_dict,
        target_syl=target_syl
    )

    catch_mean, catch_sem, all_folders = summarize_day_values(catch_day_raw_values)

    n_days = len(all_folders)
    folder_to_idx = {f: i for i, f in enumerate(all_folders)}
    
    train_folders = [f for f in all_folders if f.lower().startswith("train")]
    last3_train_folders = train_folders[-3:]
    post_folders = [f for f in all_folders if f.lower().startswith("post")]

    baseline_values = []
    for folder_name, vals in catch_day_raw_values.items():
        if folder_name.lower().startswith("base"):
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                baseline_values.append(vals)

    baseline_values = (
        np.concatenate(baseline_values)
        if len(baseline_values) > 0
        else np.array([], dtype=float)
    )

    p_last3_for_plot = np.nan
    sig_last3_for_plot = False

    if baseline_values.size == 0:
        print("[warn] No baseline values found.")
    else:
        baseline_mean_pooled = float(np.nanmean(baseline_values))
        baseline_sem_pooled = sem_of_valid(baseline_values)
        baseline_norm = normality_from_values(baseline_values)

        pooled_last3_vals = []
        for folder_name in last3_train_folders:
            vals = catch_day_raw_values.get(folder_name, np.array([], dtype=float))
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                pooled_last3_vals.append(vals)

        pooled_last3_vals = (
            np.concatenate(pooled_last3_vals)
            if len(pooled_last3_vals) > 0
            else np.array([], dtype=float)
        )

        post_vals_list = []
        for folder_name in post_folders:
            vals = catch_day_raw_values.get(folder_name, np.array([], dtype=float))
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                post_vals_list.append(vals)

        post_values = (
            np.concatenate(post_vals_list)
            if len(post_vals_list) > 0
            else np.array([], dtype=float)
        )

        last3_mean = float(np.nanmean(pooled_last3_vals)) if pooled_last3_vals.size > 0 else np.nan
        last3_sem = sem_of_valid(pooled_last3_vals)

        post_mean = float(np.nanmean(post_values)) if post_values.size > 0 else np.nan
        post_sem = sem_of_valid(post_values)

        print("\n=== Pooled Group Means ===")
        print(
            f"Baseline pooled:\n"
            f"n={baseline_values.size}, mean={baseline_mean_pooled:.4f}, sem={baseline_sem_pooled:.4f}"
        )
        print(
            f"Last 3 training days pooled:\n"
            f"n={pooled_last3_vals.size}, mean={last3_mean:.4f}, sem={last3_sem:.4f}"
        )
        print(
            f"Postbaseline pooled:\n"
            f"n={post_values.size}, mean={post_mean:.4f}, sem={post_sem:.4f}"
        )

        print("\nBaseline vs last 3 training days")

        if pooled_last3_vals.size == 0:
            print("[warn] no valid data for last 3 training days.")
        else:
            pooled_last3_norm = normality_from_values(pooled_last3_vals)

            can_ttest_last3 = (
                baseline_norm["normal"]
                and pooled_last3_norm["normal"]
                and baseline_values.size >= 2
                and pooled_last3_vals.size >= 2
            )

            if can_ttest_last3:
                test_name_last3 = "Welch t-test"
                test_last3 = stats.ttest_ind(
                    baseline_values,
                    pooled_last3_vals,
                    equal_var=False
                )
            else:
                test_name_last3 = "Mann-Whitney-U"
                test_last3 = stats.mannwhitneyu(
                    baseline_values,
                    pooled_last3_vals,
                    alternative="two-sided"
                )

            stat_last3 = float(test_last3.statistic)
            p_last3 = float(test_last3.pvalue)

            p_last3_for_plot = p_last3
            sig_last3_for_plot = bool(p_last3 < 0.05)

            sig_txt = "significant" if sig_last3_for_plot else "not significant"

            print(
                f"Comparison Baseline vs last 3 training days "
                f"Test={test_name_last3}, "
                f"Stat={stat_last3:.4f}, "
                f"p={p_last3:.4g} -> {sig_txt}"
            )

    x = np.arange(n_days)
    fig, ax = plt.subplots(figsize=figsize)

    plot_segments(
        ax,
        x,
        catch_mean,
        catch_sem,
        color_catch,
        line_zorder=3,
        fill_zorder=3
    )

    phase_changes = []
    for i in range(1, len(all_folders)):
        prev = all_folders[i - 1].lower()
        curr = all_folders[i].lower()

        if prev.startswith("base") and curr.startswith("train"):
            phase_changes.append(i - 0.5)
        elif prev.startswith("train") and curr.startswith("post"):
            phase_changes.append(i - 0.5)

    for xc in phase_changes:
        ax.axvline(x=xc, color="black", linestyle="--", lw=1, alpha=1)

    short_labels = []
    base_count = 0
    train_count = 0
    post_count = 0

    for folder in all_folders:
        folder_low = folder.lower()

        if folder_low.startswith("base"):
            base_count += 1
            short_labels.append(f"B{base_count}")
        elif folder_low.startswith("train"):
            train_count += 1
            short_labels.append(f"T{train_count}")
        elif folder_low.startswith("post"):
            post_count += 1
            short_labels.append(f"P{post_count}")
        else:
            short_labels.append(folder)

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha="right")

    ax.set_yticks([2, 3, 4, 5])
    ax.set_ylabel("Mean Repeat Number")

    legend_handles = [
        Line2D([0], [0], color=color_catch, linewidth=1, label=target_syl)
    ]
    ax.legend(handles=legend_handles, frameon=False)

    # Signifikanz-Markierung: Baseline vs letzte 3 Trainingstage
    if sig_last3_for_plot:
        base_idx_candidates = [
            i for i, f in enumerate(all_folders)
            if f.lower().startswith("base")
        ]

        last3_idx = [
            folder_to_idx[f]
            for f in last3_train_folders
            if f in folder_to_idx
        ]

        if len(base_idx_candidates) > 0 and len(last3_idx) > 0:
            x_base = float(np.mean(base_idx_candidates))
            x_train = float(np.mean(last3_idx))

            y_min, y_max = ax.get_ylim()
            y_span = y_max - y_min if np.isfinite(y_max - y_min) and y_max > y_min else 1.0

            y_bracket = y_max + 0.04 * y_span
            h_bracket = 0.02 * y_span

            stars = stars_from_p(p_last3_for_plot)

            ax.plot(
                [x_base, x_base, x_train, x_train],
                [y_bracket, y_bracket + h_bracket, y_bracket + h_bracket, y_bracket],
                color="black",
                linewidth=1,
                zorder=6
            )

            ax.text(
                (x_base + x_train) / 2,
                y_bracket + h_bracket + 0.01 * y_span,
                stars,
                ha="center",
                va="bottom",
                color="black",
                zorder=7
            )

            ax.set_ylim(y_min, y_max + 0.12 * y_span)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=2)

    fig.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        if save_name is None:
            safe_target = re.sub(r"[^A-Za-z0-9]+", "_", target_syl).strip("_")
            save_name = f"mean_repeat_length_{safe_target}.png"

        save_path = os.path.join(save_dir, save_name)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()

    return catch_mean, all_folders

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

def summary_mean_repeat_number_across_birds(
    birds,
    target_syl_per_bird,
    context_per_bird,
    bird_name_mapping,
    n_last_training_days=3,
    bird_colors=None,
    figsize=(6, 6),
    save_dir=None,
    save_name=None,
    dpi=300,
):
    """
    Summary plot for mean repeat number in baseline and training for mulitple 
    experiments, including mean + SEM.
    
    Parameters
    ----------
    birds : dict
        Dictionary of multiple birds/ experiments 
    target_syl_per_bird: dict
        Target syllable for every experiment, ordered by experiment name
    context_per_bird : dict [bool]
        True depicts a context dependency of the target syllable (e.g. 'kB' only
        counts 'B' preceded by 'k'). Individual value for each target syllable
    bird_name_mapping: dict
        Maps experiment names to more concise letters for the legend
    n_last_training_days: int
        Number of last training days to consider,
        default : 3
        
    Returns
    -------
    Dict :
        values per bird, including baseline/ training values, relative change,
        group mean + SEM, results of Wilcoxon tests.
    
    """

    bird_names = []
    baseline_vals = []
    training_vals = []

    for bird, labels_dict in birds.items():
        target_syl = target_syl_per_bird[bird]
        context = context_per_bird[bird]

        day_data = compute_day_raw_values(
            labels_dict=labels_dict,
            target_syl=target_syl,
            context=context
        )

        baseline_days = [
            (day_name, vals)
            for day_name, vals in day_data.items()
            if day_name.lower().startswith("base")
        ]

        training_days = [
            (day_name, vals)
            for day_name, vals in day_data.items()
            if day_name.lower().startswith("train")
        ]

        training_days = sorted(training_days, key=lambda x: day_sort_key(x[0]))
        
        if n_last_training_days is not None:
            training_days = training_days[-n_last_training_days:]

        baseline_lengths = pool_all_lengths(baseline_days)
        training_lengths = pool_all_lengths(training_days)
        
        base_mean = nanmean_or_nan(baseline_lengths) 
        train_mean = nanmean_or_nan(training_lengths)
        
        bird_names.append(bird)
        baseline_vals.append(base_mean)
        training_vals.append(train_mean)

    baseline_vals = np.array(baseline_vals, dtype=float)
    training_vals = np.array(training_vals, dtype=float)

    relative_to_baseline_vals = np.full(len(bird_names), np.nan, dtype=float)
    valid_base = np.isfinite(baseline_vals) & (baseline_vals != 0)
    relative_to_baseline_vals[valid_base] = training_vals[valid_base] / baseline_vals[valid_base]
    percent_change_vals = (relative_to_baseline_vals - 1.0) * 100.0

    display_bird_names = [map_bird_name(bird, bird_name_mapping) for bird in bird_names]

    print("\nTraining/Baseline raw values and % change per repeat:")
    for i, name in enumerate(display_bird_names):
        base_raw = baseline_vals[i]
        train_raw = training_vals[i]
        if np.isfinite(percent_change_vals[i]):
            print(
                f"  {name}: baseline={base_raw:.4f}, training={train_raw:.4f}, "
                f"delta={percent_change_vals[i]:+.2f}%"
            )
        else:
            base_txt = f"{base_raw:.4f}" if np.isfinite(base_raw) else "nan"
            train_txt = f"{train_raw:.4f}" if np.isfinite(train_raw) else "nan"
            print(
                f"  {name}: baseline={base_txt}, training={train_txt}, "
                f"delta=n/a (Baseline fehlt oder = 0)"
            )

    group_pct_mean = nanmean_or_nan(percent_change_vals)
    group_pct_sem = sem_of_valid(percent_change_vals)
    print(
        f"Group percent change: mean={group_pct_mean:+.2f}% ± SEM={group_pct_sem:.2f}%"
    )

    # --- Wilcoxon signed-rank test: baseline vs training ---
    valid_mask = np.isfinite(baseline_vals) & np.isfinite(training_vals)
    baseline_valid = baseline_vals[valid_mask]
    training_valid = training_vals[valid_mask]

    wilcoxon_stat = np.nan
    wilcoxon_p = np.nan
    wilcoxon_n = len(baseline_valid)

    if wilcoxon_n >= 2:
        diffs = training_valid - baseline_valid
        n_nonzero = np.sum(diffs != 0)

        if n_nonzero > 0:
            try:
                res = stats.wilcoxon(
                    baseline_valid,
                    training_valid,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto"
                )
                wilcoxon_stat = float(res.statistic)
                wilcoxon_p = float(res.pvalue)
            except ValueError as e:
                print(f"Wilcoxon test could not be computed: {e}")
        else:
            print("Wilcoxon test not computed: all paired differences are exactly zero.")
    else:
        print("Wilcoxon test not computed: no valid pairs.")

    print(
        f"Wilcoxon signed-rank test (baseline vs training): "
        f"n={wilcoxon_n}, statistic={wilcoxon_stat}, p={wilcoxon_p}"
    )

    fig, ax = plt.subplots(figsize=figsize)
    x_base_pos, x_train_pos = setup_bs_t_summary_axis(ax, 1, 1)

    x_base = np.full(len(bird_names), x_base_pos)
    x_train = np.full(len(bird_names), x_train_pos)

    if bird_colors is None:
        bird_colors = {}
    wes_palette = [
        "#3B9AB2",
        "#78B7C5",
        "#E1AF00",
        "#EBCC2A",
        "#F21A00",
        "#9E7A7A",
        "#5F9D8A",
        "#C7B19C",
    ]
    colors = [
        bird_colors.get(bird, wes_palette[i % len(wes_palette)])
        for i, bird in enumerate(bird_names)
    ]

    for i, bird in enumerate(bird_names):
        if np.isnan(baseline_vals[i]) and np.isnan(training_vals[i]):
            continue

        ax.plot(
            [x_base[i], x_train[i]],
            [baseline_vals[i], training_vals[i]],
            color=colors[i],
            lw=1,
            alpha=0.9,
            zorder=1,
            label=display_bird_names[i]
        )

    # --- Group mean + SEM ---
    group_base = nanmean_or_nan(baseline_vals)
    group_train = nanmean_or_nan(training_vals)
    group_base_sem = sem_of_valid(baseline_vals)
    group_train_sem = sem_of_valid(training_vals)

    ax.errorbar(
        [x_base_pos, x_train_pos],
        [group_base, group_train],
        yerr=[group_base_sem, group_train_sem],
        color="black",
        lw=1.2,
        marker="o",
        markersize=4,
        capsize=3,
        elinewidth=1.0,
        alpha=1,
        zorder=4,
        #label="Group mean ± SEM"
    )

    ax.set_ylabel("Mean Repeat Number")
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "","3", "", "5"])
    ax.set_ylim(1, )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- p-value / significance annotation ---
    y = 1.02
    h = 0.03

    ax.plot(
        [x_base_pos, x_base_pos, x_train_pos, x_train_pos],
        [y, y + h, y + h, y],
        transform=ax.get_xaxis_transform(),
        color="black",
        lw=1,
        clip_on=False
    )

    p_text = stars_from_p(wilcoxon_p)

    ax.text(
        (x_base_pos + x_train_pos) / 2,
        y + h,
        p_text,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        clip_on=False
    )

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
        handlelength=0.6
    )

    finalize_bs_t_summary_layout(fig)
    pos = ax.get_position()
    scale = 0.69
    ax.set_position([
        pos.x0 + pos.width * (1 - scale) / 2,
        pos.y0 + pos.height * (1 - scale) / 2,
        pos.width * scale,
        pos.height * scale
    ])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            save_name = "summary_mean_repeat_length_across_birds.png"
        save_path = os.path.join(save_dir, save_name)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()

    print(f"baseline mean: {group_base:.4f} ± {group_base_sem:.4f}")
    print(f"training mean: {group_train:.4f} ± {group_train_sem:.4f}")
    print(f"relative change: {group_pct_mean:+.2f}%")

    return {
        "birds": bird_names,
        "baseline_means": baseline_vals,
        "training_means": training_vals,
        "relative_to_baseline": relative_to_baseline_vals,
        "percent_change": percent_change_vals,
        "group_percent_change_mean": group_pct_mean,
        "group_percent_change_sem": group_pct_sem,
        "group_baseline_mean": group_base,
        "group_training_mean": group_train,
        "group_baseline_sem": group_base_sem,
        "group_training_sem": group_train_sem,
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_pvalue": wilcoxon_p,
        "wilcoxon_n_pairs": wilcoxon_n
    }

def summary_selftransition_relative_around_target(
    birds,
    target_syl_per_bird,
    target_repeat_per_bird,
    context_per_bird,
    n_last_training_days=3,
    offsets=(-1, 0, 1),
    bird_colors=None,
    bird_name_mapping=None,
    figsize=(6, 5),
    save_dir=None,
    save_name=None,
    dpi=300,
    min_support_files=3
):
    """
    Self-transition probability of each repeat number within a repeat phrase
    for each experiment in training relative to baseline.

    Parameters
    ----------
    birds : dict
        Dictionary of multiple birds/ experiments 
    target_syl_per_bird: dict [str]
        Target syllable for every experiment, ordered by experiment name
    target_repeat_per_bird : dict [int]
        Targeted repeat number for experiment, ordered by experiment name
        For branch point experiments, this is None
    context_per_bird : dict [bool]
        True depicts a context dependency of the target syllable (e.g. 'kB' only
        counts 'B' preceded by 'k'). Individual value for each target syllable
    bird_name_mapping: dict
        Maps experiment names to more concise letters for the legend
    bird_colors : dict
        Maps a specific color to each experiment
    n_last_training_days: int
        Number of last training days to consider,
        default : 3
    offsets : list of int
        offsets relative to target repeat number that should be plotted,
        default : (-1, 0, 1)
    min_support_files : int 
        minimum number of files needed to show a data point,
        default : 3

    Returns
    -------
        dict:
            'offsets'        
                list of plottet offset labels
            'per_bird'
                per experiment, baseline and training mean, delta, valid data
    """

    if not isinstance(birds, dict) or not all(hasattr(v, "items") for v in birds.values()):
        raise TypeError(
            "birds must be a multi-bird dictionary: {bird_name: labels_dict}. "
            "Single-bird input as a plain labels_dict is no longer supported."
        )

    def resolve_per_bird_arg(arg, bird_name, arg_name):
        if isinstance(arg, dict):
            if bird_name in arg:
                return arg[bird_name]
            if len(arg) == 1:
                return next(iter(arg.values()))
            raise KeyError(
                f"'{arg_name}' missing entry for bird '{bird_name}'. "
                f"Existing keys: {list(arg.keys())[:5]}"
            )
        return arg

    if bird_colors is None:
        bird_colors = {}
    if bird_name_mapping is None:
        bird_name_mapping = {}

    wes_palette = [
        "#3B9AB2",
        "#78B7C5",
        "#E1AF00",
        "#EBCC2A",
        "#F21A00",
        "#9E7A7A",
        "#5F9D8A",
        "#C7B19C",
    ]

    birds_order = list(birds.keys())
    bird_color_map = {
        bird: bird_colors.get(bird, wes_palette[i % len(wes_palette)])
        for i, bird in enumerate(birds_order)
    }

    offset_labels = list(offsets)

    per_bird_details = {}

    for bird, labels_dict in birds.items():
        target_syl = resolve_per_bird_arg(target_syl_per_bird, bird, "target_syl_per_bird")
        target_repeat = int(resolve_per_bird_arg(target_repeat_per_bird, bird, "target_repeat_per_bird"))
        context = resolve_per_bird_arg(context_per_bird, bird, "context_per_bird")

        per_bird_details[bird] = {}

        for off in offset_labels:
            # current repeat number 
            repeat_number_for_position = target_repeat + off + 1

            # repeat number = 1 is invalid
            if repeat_number_for_position < 2:
                per_bird_details[bird][off] = {
                    "baseline_mean": np.nan,
                    "training_mean": np.nan,
                    "delta_pp": np.nan,
                    "repeat_number_call": repeat_number_for_position,
                    "n_valid_baseline_days": 0,
                    "n_valid_training_days": 0,
                }
                continue

            all_training_days = sorted(
                {
                    os.path.basename(os.path.dirname(p))
                    for p in labels_dict.keys()
                    if os.path.basename(os.path.dirname(p)).lower().startswith("train")
                },
                key=day_sort_key,
            )

            if n_last_training_days is not None:
                all_training_days = all_training_days[-n_last_training_days:]
            training_day_set = set(all_training_days)

            base_mean, n_base_events, n_base_files, n_base_files_with_events = pooled_p_self(
                labels_dict,
                target_syl=target_syl,
                repeat_number=repeat_number_for_position,
                context=context,
                day_predicate=lambda d: d.lower().startswith("base"),
            )
            train_mean, n_train_events, n_train_files, n_train_files_with_events = pooled_p_self(
                labels_dict,
                target_syl=target_syl,
                repeat_number=repeat_number_for_position,
                context=context,
                day_predicate=lambda d: d in training_day_set,
            )

            if not np.isfinite(base_mean) or not np.isfinite(train_mean):
                delta_pp = np.nan
            elif np.isclose(base_mean, 0.0) and np.isclose(train_mean, 0.0):
                delta_pp = np.nan
            else:
                delta_pp = train_mean - base_mean

            per_bird_details[bird][off] = {
                "baseline_mean": base_mean,
                "training_mean": train_mean,
                "delta_pp": delta_pp,
                "repeat_number_call": repeat_number_for_position,
                "n_valid_baseline_days": int(len({
                    os.path.basename(os.path.dirname(p))
                    for p in labels_dict.keys()
                    if os.path.basename(os.path.dirname(p)).lower().startswith("base")
                })),
                "n_valid_training_days": int(len(training_day_set)),
                "n_events_baseline": int(n_base_events),
                "n_events_training": int(n_train_events),
                "n_files_baseline": int(n_base_files),
                "n_files_training": int(n_train_files),
                "n_files_with_events_baseline": int(n_base_files_with_events),
                "n_files_with_events_training": int(n_train_files_with_events),
            }

    fig, ax = plt.subplots(figsize=figsize)

    x_positions = np.arange(len(offset_labels))
    
    def is_plottable_delta(v, n_files_with_events_training=0):
        if not np.isfinite(v):
            return False
        if int(n_files_with_events_training) < int(min_support_files):
            return False
        return True

    if 0 in offset_labels:
        i0 = offset_labels.index(0)
        ax.axvspan(
            x_positions[i0] - 0.5,
            x_positions[-1] + 0.5,
            color="lightgrey",
            alpha=0.35,
            zorder=0,
            linewidth=0,
        )

    box_data_clean = []
    for off in offset_labels:
        valid_vals = []
        for bird in birds_order:
            details = per_bird_details[bird][off]
            delta_pp = details["delta_pp"]
            if is_plottable_delta(
                delta_pp,
                details.get("n_files_with_events_training", 0),
            ):
                valid_vals.append(float(delta_pp))
        box_data_clean.append(np.asarray(valid_vals, dtype=float))

    bp = ax.boxplot(
        box_data_clean,
        positions=x_positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 0.6, "zorder": 4},
        whiskerprops={"linewidth": 0.6, "color": "black", "zorder": 3},
        capprops={"linewidth": 0.6, "color": "black", "zorder": 3},
        boxprops={"linewidth": 0.6, "color": "black", "zorder": 3},
    )

    for box in bp["boxes"]:
        box.set_facecolor("none")
        box.set_alpha(1.0)
        box.set_zorder(3)

    line_points_by_bird = {bird: [] for bird in birds_order}
    for i, off in enumerate(offset_labels):
        for bird in birds_order:
            details = per_bird_details[bird][off]
            delta_pp = details["delta_pp"]
            if not is_plottable_delta(
                delta_pp,
                details.get("n_files_with_events_training", 0),
            ):
                continue

            x_j = x_positions[i]
            line_points_by_bird[bird].append((x_j, float(delta_pp)))

    for bird in birds_order:
        pts = line_points_by_bird[bird]
        if len(pts) < 2:
            continue
        pts = sorted(pts, key=lambda t: t[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs,
            ys,
            color=bird_color_map[bird],
            linewidth=0.5,
            zorder=2,
        )

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=2,
            markerfacecolor=bird_color_map[bird],
            markeredgecolor=bird_color_map[bird],
            label=map_bird_name(bird, bird_name_mapping),
        )
        for bird in birds_order
    ]
    if len(legend_handles) > 0:
        ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            borderaxespad=0.0,
        )

    if any(len(arr) > 0 for arr in box_data_clean):
        ax.set_ylim(-0.26, 0.53)
        ax.set_yticks([-0.25, 0.0, 0.25, 0.5])

    mean_delta_by_offset = {}

    print("\n=== Mean delta per offset ===")

    y_low, y_high = ax.get_ylim()
    y_span = y_high - y_low
    y_text = y_high + 0.08 * y_span
    for i, vals in enumerate(box_data_clean):
        off = offset_labels[i]

        if len(vals) == 0:
            mean_delta_by_offset[off] = np.nan
            print(f"offset {off:+d}: mean delta = nan, n=0")
            continue

        mean_delta = float(np.mean(vals))
        mean_delta_by_offset[off] = mean_delta

        print(
            f"offset {off:+d}: "
            f"mean delta = {mean_delta:.4f} "
            f"({mean_delta * 100:.1f} percentage points), "
            f"n={len(vals)}"
        )
        ax.text(
            x_positions[i],
            y_text,
            f"△={mean_delta:.2f}",
            rotation=45,
            ha="center",
            va="bottom",
            color="black",
            zorder=5,
            clip_on=False,
        )

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.9)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [f"{off:+d}".replace("+0", "0") for off in offset_labels]
    )
    ax.tick_params(axis="y")
    ax.set_ylabel("Δ Self-Transition Prob.")
    ax.set_xlabel("Rel. Position to Target")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    scale = 0.68
    pos = ax.get_position()

    ax.set_position([
        pos.x0 + 0.05, 
        pos.y0 + (pos.height * 0.2),
        pos.width * scale,
        pos.height * scale
    ])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            save_name = "summary_selftransition_relative_around_target_boxplot.png"
        save_path = os.path.join(save_dir, save_name)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()

    return {
        "offsets": offset_labels,
        "per_bird": per_bird_details
    }

def metarepeat_transitions(
    labels_dict,
    target_syl,
    context=False,
    n_last_training_days=3,
    max_position=None,
    figsize=(1.9, 2.9),
    save_dir=None,
    save_name=None,
    dpi=300,
    train_color="#1b35cfff",
    marker="o",
    markersize=2,
    linewidth=1,
    missing_value=0.0,
):
    """
    Combined plot of within- and between-chunk transition probability of a meta-repeat
    during baseline and training

    Parameters
    ----------
    labels_dict : dict
        Dictionary containing label sequences over one or multiple days
    target_syl : str
        Syllable for which the repeat number should be calculated
    context : bool
        if True, count context-dependent repeats, 
        thus only counts repeating syllable in the preceding context
        (e.g. "kb" only counts b+ with k preceding)
    n_last_training_days : int
        Number of last training days to consider,
        default : 3
    max_position : int
        Limits the number of repeat positions to be considered,
        upper limit, default : None, thus all positions are used
    missing_value : float
        Value used for missing transition probabilities at positions 
        where one phase has no data,
        default : 0.0

    Returns
    -------
        Dict :
            repeat unit, baseline/training stats, training days used and figure
    """

    repeat_unit = target_syl[3:-2].rstrip("+")
    if len(repeat_unit) != 2:
        raise ValueError(
            "Currently only usable for two-sign repeat units, "
            "z.B. target_syl='(?:dc)+'."
        )

    first_char, second_char = repeat_unit[0], repeat_unit[1]
    unit_len = len(repeat_unit)

    baseline_labels = {
        path: seq 
        for path, seq in labels_dict.items()
        if os.path.basename(os.path.dirname(path)).lower().startswith("base")
    }

    training_days = sorted(
        {
            os.path.basename(os.path.dirname(path))
            for path in labels_dict.keys()
            if os.path.basename(os.path.dirname(path)).lower().startswith("train")
        },
        key=day_sort_key,
    )
    if n_last_training_days is not None:
        training_days = training_days[-int(n_last_training_days):]
    training_day_set = set(training_days)

    training_labels = {
        path: seq
        for path, seq in labels_dict.items()
        if os.path.basename(os.path.dirname(path)) in training_day_set
    }

    # Counts und Wahrscheinlichkeiten getrennt fuer Baseline und Training berechnen.
    phase_stats = {}
    for phase_name, labels_subset in [
        ("baseline", baseline_labels),
        ("training", training_labels),
    ]:
        first_to_second_transition = defaultdict(int)
        first_to_second_other = defaultdict(int)
        second_to_first_transition = defaultdict(int)
        second_to_first_other = defaultdict(int)
        n_matches = 0

        for _, seq in labels_subset.items():
            if not isinstance(seq, str) or not seq:
                continue

            for match in re.finditer(target_syl, seq):
                full_match = match.group(0)

                if context:
                    if len(full_match) <= 1:
                        continue
                    repeat_part = full_match[1:]
                    repeat_start_idx = match.start() + 1
                else:
                    repeat_part = full_match
                    repeat_start_idx = match.start()

                repeat_len = len(repeat_part) // unit_len
                if repeat_len <= 0:
                    continue

                n_matches += 1

                for pos in range(1, repeat_len + 1):
                    if max_position is not None and pos > int(max_position):
                        continue

                    first_idx = repeat_start_idx + (pos - 1) * unit_len
                    second_idx = first_idx + 1

                    # first_n -> second_n, z.B. d_n -> c_n
                    if first_idx + 1 < len(seq) and seq[first_idx + 1] == second_char:
                        first_to_second_transition[pos] += 1
                    else:
                        first_to_second_other[pos] += 1

                    # second_n -> first_{n+1}, z.B. c_n -> d_{n+1}
                    if second_idx + 1 < len(seq) and seq[second_idx + 1] == first_char:
                        second_to_first_transition[pos] += 1
                    else:
                        second_to_first_other[pos] += 1

        phase_stats[phase_name] = {
            "first_to_second": {},
            "second_to_first_next": {},
            "n_matches": int(n_matches),
        }

        for key, trans_counts, other_counts in [
            ("first_to_second", first_to_second_transition, first_to_second_other),
            ("second_to_first_next", second_to_first_transition, second_to_first_other),
        ]:
            all_positions = sorted(set(trans_counts.keys()) | set(other_counts.keys()))
            for pos in all_positions:
                n_transition = int(trans_counts[pos])
                n_other = int(other_counts[pos])
                n_total = n_transition + n_other
                p_transition = n_transition / n_total if n_total > 0 else np.nan
                phase_stats[phase_name][key][pos] = {
                    "n_transition": n_transition,
                    "n_other": n_other,
                    "n_total": n_total,
                    "p_transition": p_transition,
                }

    baseline_stats = phase_stats["baseline"]
    training_stats = phase_stats["training"]

    # Tests berechnen und optional direkt in die Konsole printen.
    tests_out = {}
    print("\n=== TWO-CHAR TRANSITION TESTS ===")
    print(f"target_syl={target_syl}, repeat_unit={repeat_unit}, context={context}")
    print(f"training_days_used={training_days}")
    print(
        f"matches: baseline={baseline_stats.get('n_matches', 0)}, "
        f"training={training_stats.get('n_matches', 0)}"
    )

    # --- Mean delta summaries ---
    delta_summary = {}

    print("\n=== MEAN DELTA SUMMARY ===")

    for key, label in [
        ("first_to_second", f"{first_char}->{second_char}"),
        ("second_to_first_next", f"{second_char}->{first_char}"),
    ]:
        base_pos_stats = baseline_stats.get(key, {})
        train_pos_stats = training_stats.get(key, {})

        positions = sorted(set(base_pos_stats.keys()) | set(train_pos_stats.keys()))
        if max_position is not None:
            positions = [p for p in positions if p <= int(max_position)]

        per_position = {}
        finite_deltas = []
        finite_percent_changes = []

        print(f"\n{label}:")

        for pos in positions:
            b = base_pos_stats.get(pos, {})
            t = train_pos_stats.get(pos, {})

            p_base = b.get("p_transition", np.nan)
            p_train = t.get("p_transition", np.nan)

            if np.isfinite(p_base) and np.isfinite(p_train):
                delta = p_train - p_base
                finite_deltas.append(delta)

                if np.isclose(p_base, 0.0):
                    percent_delta_change = np.nan
                else:
                    percent_delta_change = (delta / p_base) * 100

                if np.isfinite(percent_delta_change):
                    finite_percent_changes.append(percent_delta_change)
            else:
                delta = np.nan
                percent_delta_change = np.nan

            per_position[pos] = {
                "p_baseline": float(p_base) if np.isfinite(p_base) else np.nan,
                "p_training": float(p_train) if np.isfinite(p_train) else np.nan,
                "delta": float(delta) if np.isfinite(delta) else np.nan,
                "percent_delta_change": (
                    float(percent_delta_change)
                    if np.isfinite(percent_delta_change)
                    else np.nan
                ),
            }

            p_base_txt = f"{p_base:.4f}" if np.isfinite(p_base) else "nan"
            p_train_txt = f"{p_train:.4f}" if np.isfinite(p_train) else "nan"
            delta_txt = f"{delta:.4f}" if np.isfinite(delta) else "nan"

            if np.isfinite(percent_delta_change):
                pct_txt = f"{percent_delta_change:.1f}%"
            else:
                pct_txt = "nan"

            print(
                f"  pos {pos:>2}: "
                f"baseline={p_base_txt}, "
                f"training={p_train_txt}, "
                f"delta={delta_txt}, "
                f"% delta change={pct_txt}"
            )

        mean_delta = (
            float(np.mean(finite_deltas))
            if len(finite_deltas) > 0
            else np.nan
        )

        mean_percent_delta_change = (
            float(np.mean(finite_percent_changes))
            if len(finite_percent_changes) > 0
            else np.nan
        )

        print(
            f"  mean delta over positions = "
            f"{mean_delta:.4f}"
            if np.isfinite(mean_delta)
            else "  mean delta over positions = nan"
        )

        if np.isfinite(mean_percent_delta_change):
            print(
                f"  mean % delta change over positions = "
                f"{mean_percent_delta_change:.1f}%"
            )
        else:
            print("  mean % delta change over positions = nan")

        delta_summary[key] = {
            "per_position": per_position,
            "mean_delta": mean_delta,
            "mean_percent_delta_change": mean_percent_delta_change,
        }

    # Combined plot.
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
    )

    for ax, key, ylabel, show_xlabel, show_legend in [
        (ax_top, "first_to_second", f"p({first_char}ₙ→{second_char}ₙ)", False, True),
        (ax_bottom, "second_to_first_next", f"p({second_char}ₙ→{first_char}ₙ₊₁)", True, False),
    ]:
        positions = sorted(set(baseline_stats[key].keys()) | set(training_stats[key].keys()))
        if max_position is not None:
            positions = [p for p in positions if p <= int(max_position)]
        positions = np.asarray(positions, dtype=int)

        if len(positions) == 0:
            ax.set_visible(False)
            continue

        y_base = np.asarray(
            [baseline_stats[key].get(int(pos), {}).get("p_transition", missing_value) for pos in positions],
            dtype=float,
        )
        y_train = np.asarray(
            [training_stats[key].get(int(pos), {}).get("p_transition", missing_value) for pos in positions],
            dtype=float,
        )

        ax.plot(
            positions,
            y_base,
            marker=marker,
            markersize=markersize,
            color="black",
            linewidth=linewidth,
            label="B",
        )
        ax.plot(
            positions,
            y_train,
            marker=marker,
            markersize=markersize,
            color=train_color,
            linewidth=linewidth,
            label="T",
        )

        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xticks(positions)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if show_xlabel:
            ax.set_xlabel("Repeat Position n")
        else:
            ax.tick_params(axis="x", labelbottom=False)

        if show_legend:
            ax.legend(frameon=False, loc="upper right", handlelength=1.0)

    fig.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        name = save_name or f"transition_positions_{first_char}{second_char}"
        save_path = os.path.join(save_dir, f"{name}.svg")
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()

    return {
        "repeat_unit": repeat_unit,
        "baseline": baseline_stats,
        "training": training_stats,
        "tests": tests_out,
        "training_days_used": training_days,
        "fig": fig,
    }

#----- Run Main -----#

if __name__ == "__main__":

    # Fig 2B
    plot_repeat_distributions_per_phase(
        labels_dict=birds_catch["Bird2_J6"],
        target_syl="j+",
        color="#4d80ffff",
        figsize=(1.9, 1.5),
        save_dir=get_figure_dir("Figure2"),
        save_name="Fig2_B.svg",
        dpi=300
    )

    # Fig 2C
    plot_mean_repeat_number_fullplot(
        labels_dict=birds_catch["Bird2_J6"],
        target_syl="j+",
        color_catch="#4d80ffff",
        figsize=(1.9, 1.5),
        save_dir=get_figure_dir("Figure2"),
        save_name="Fig2_C.svg",
        dpi=300
    )

    # Fig 2D
    summary_mean_repeat_number_across_birds(
        birds = {
            bird: data
            for bird, data in birds_catch.items()
            if bird != "Bird1_bkd"
        },
        target_syl_per_bird = {
            bird : data
            for bird, data in target_syl_per_bird.items()
            if bird != "Bird1_bkd"
        },
        context_per_bird = {
            bird : data
            for bird, data in context_per_bird.items()
            if bird != "Bird1_bkd"
        },
        bird_name_mapping = bird_name_mapping,
        n_last_training_days=3,
        bird_colors=bird_colors,
        figsize=(1.9, 1.5),
        save_dir=get_figure_dir("Figure2"),
        save_name="Fig2_D.svg",
        dpi=300
    )

    # Fig 2E
    summary_rel = summary_selftransition_relative_around_target(
        birds= {bird : data
                for bird, data in birds_catch.items()
                if bird != "Bird1_bkd"},
        target_syl_per_bird= {bird : data
                for bird, data in target_syl_per_bird.items()
                if bird != "Bird1_bkd"},
        target_repeat_per_bird= {bird : data
                for bird, data in target_repeat_per_bird.items()
                if bird != "Bird1_bkd"},
        context_per_bird= {bird : data
                for bird, data in context_per_bird.items()
                if bird != "Bird1_bkd"},
        offsets=(-5, -4, -3, -2, -1, 0, 1, 2, 3),
        bird_colors=bird_colors,
        bird_name_mapping=bird_name_mapping,
        figsize=(1.9, 1.5),
        save_dir=get_figure_dir("Figure2"),
        save_name="Fig2_E.svg",
        dpi=300
    )

    # statistic for Fig 2E
    wilcox_df, wilcox_details = wilcoxon_by_training_from_summary(
        summary_rel,
        min_support_files=3
    )
    pvals = wilcox_df["pvalue"].to_numpy()
    mask = np.isfinite(pvals)
    pvals_corr = np.full_like(pvals, np.nan, dtype=float)
    if mask.sum() > 0:
        _, pvals_corr_valid, _, _ = multipletests(pvals[mask], method="holm")
        pvals_corr[mask] = pvals_corr_valid
    wilcox_df["pvalue_holm"] = pvals_corr
    print(wilcox_df)

    # Fig 2G
    plot_repeat_distributions_per_phase(
        labels_dict=birds_catch["Bird2_DC4"],
        target_syl="(?:dc)+",
        color="#1b35cfff",
        figsize=(1.9, 1.5),
        save_dir=get_figure_dir("Figure2"),
        save_name="Fig2_G.svg",
        dpi=300
    )

    # Fig 2H
    metarepeat_transitions(
        labels_dict=birds_catch["Bird2_DC4"],
        target_syl="(?:dc)+",
        context=False,
        n_last_training_days=3,
        figsize=(1.9, 1.5),
        save_dir=get_figure_dir("Figure2"),
        save_name="Fig2_H.svg",
        dpi=300
    )