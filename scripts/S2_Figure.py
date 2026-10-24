"""
Code for results depicted in Supplementary Figure 2
    A) Transition probability plot over days for Control 1
    B) Repeat distributions for baseline and training for Control 2
    C) Number of syllables after the target in baseline vs. non-catch trials
       Number of target phrases per bout in baseline vs. training
       Appearance of the target repeat number per target phrase in baseline vs. training


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
from scipy import stats
from statsmodels.stats.multitest import multipletests
from collections import defaultdict

from util.helper_fct import (
    load_birds,
    get_figure_dir,
    CONTROL_BIRDS,
    BIRD_COLORS as bird_colors,
    BIRD_NAME_MAPPING as bird_name_mapping,
    TARGET_SYL_PER_CONTROL_BIRD as target_syl_per_control_bird,
    TARGET_REPEAT_PER_CONTROL_BIRD as target_repeat_per_control_bird,
    CONTEXT_PER_CONTROL_BIRD as context_per_control_bird
)

from util.helper_fct import (
    compute_branch_probs, plot_segments, run_2x2_branch_test, 
    add_sig_bracket, stars_from_p, count_branch_events_for_folders,
    sem_of_valid, normality_from_values, get_repeat_lengths, 
    day_sort_key, nanmean_or_nan, map_bird_name,
    finalize_bs_t_summary_layout, split_baseline_training_days, 
    setup_bs_t_summary_axis, compute_bouts_and_targets_per_day,
    compute_day_raw_values, get_lengths_from_vals,
    flatten_repeats, extract_seq_counts
    )

# ----- Load data ----- #

control_birds_catch = load_birds(source="control", catch=True)

control_colors = {
    bird: bird_colors[bird]
    for bird in CONTROL_BIRDS
}

save_dir = get_figure_dir("FigureS2")

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

def plot_branchpoint_probabilities_fullplot(
    labels_dict,
    target_syl,
    branch_syl,
    target_branch=None,
    target_line_color="#7428B3",
    other_line_color="#919191",
    show_postbase=True,
    figsize=(9, 5),
    save_dir=None,
    save_name=None,
    dpi=300
    ):
    """
    Plots branchpoint probabilities per day, separated by
    baseline, training and post-training phases.
    
    Parameters
    ---------- 
        labels_dict : dict
            Dictionary containing label sequences over one or multiple days
        target_syl : str
            the branch point syllable
        branch_syl : list[str]
            array of possible branching syllables
            transitions to syllables not listed in `branch_syl` are pooled
            into an additional "other" category for probability estimation,
            but are not shown in the plot
        target_branch : str 
            targeted syllable of the branching syllables, highlighted in 
            target_line_color and tested in 2x2 tests.
        other_line_color : str
            color for non-target branches
        show_postbase : bool
            if False, post-training days are excluded from the plot
            
    Returns
    -------
        catch_matrix_full : np.ndarray
            per-day branch probabilities for catch trial songs
        all_folders : list [str]
            ordered list of all day folders included in the analysis
        catch_stats : dict or None
            results of the 2x2 test comparing baseline vs last three training days 
    """

    # Compute branch probabilities
    catch_matrix, catch_sem, catch_folders = compute_branch_probs(
        labels_dict, target_syl, branch_syl, show_postbase=show_postbase)

    print("Catch trials per day:")
    for day in catch_folders:
        n = sum(1 for p in labels_dict if os.path.basename(os.path.dirname(p)) == day)
        print(f"  {day}: {n} file(s)")

    # Print total branchpoint events and target branch events for each day
    if target_branch is not None and target_branch in branch_syl:
        target_idx = branch_syl.index(target_branch)
        print("\nBranchpoint events per day:")
        for day in catch_folders:
            counts = count_branch_events_for_folders(
                labels_dict=labels_dict,
                folders=[day],
                target_syl=target_syl,
                branch_syl=branch_syl
            )
            n_target = int(counts[target_idx])
            n_total = int(counts.sum())
            if n_total > 0:
                percentage = 100 * n_target / n_total
                print(
                    f"  {day}: {n_target}/{n_total} target branches "
                    f"({percentage:.2f}%)"
                )
            else:
                print(f"  {day}: 0/0 target branches (no branchpoint events)")

    all_folders = catch_folders
    n_days = len(all_folders)
    folder_to_idx = {f: i for i, f in enumerate(all_folders)}

    catch_matrix_full = catch_matrix
    catch_sem_full = catch_sem

    x = np.arange(n_days)

    if target_branch is None:
        colors = [other_line_color] * len(branch_syl)
        if len(colors) > 0:
            colors[-1] = target_line_color
    else:
        colors = [target_line_color if b == target_branch else other_line_color for b in branch_syl]

    fig, ax = plt.subplots(figsize=figsize)

    for j, branch in enumerate(branch_syl):
        y_c = catch_matrix_full[:, j]
        yerr_c = catch_sem_full[:, j]
        is_target_branch = (branch == target_branch) if target_branch is not None else (j == len(branch_syl) - 1)
        plot_segments(
            ax,
            x,
            y_c,
            yerr_c,
            colors[j],
            linestyle="-",
            lw=1,
            fill_alpha=0.25,
            label=f"{target_syl}→{branch}",
            line_zorder=8 if is_target_branch else 4,
            fill_zorder=7 if is_target_branch else 3,
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
        ax.axvline(x=xc, color="black", linestyle="--")

    # X-Labels
    short_labels = []
    base_count = 0
    train_count = 0
    post_count = 0
    for folder in all_folders:
        fl = folder.lower()
        if fl.startswith("base"):
            base_count += 1
            short_labels.append(f"B{base_count}")
        elif fl.startswith("train"):
            train_count += 1
            short_labels.append(f"T{train_count}")
        elif fl.startswith("post"):
            post_count += 1
            short_labels.append(f"P{post_count}")
        else:
            short_labels.append(folder)

    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "", "0.5", "", "1"])
    ax.set_xticks(range(n_days))
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_ylabel("Transition\nProbability")

    t1_idx = None
    for idx, folder in enumerate(all_folders):
        if folder.lower().startswith("train"):
            t1_idx = idx
            break
    if t1_idx is not None:
        x0 = t1_idx + 0.08
        x1 = t1_idx + 0.28
        y_top = 0.52
        y_bottom = 0.48

        other_branches = [b for b in branch_syl if b != target_branch]
        other_branch_label = "/".join(other_branches) if other_branches else "other"
        ax.plot([x0, x1], [y_top, y_top], color=target_line_color, lw=1, zorder=10)
        ax.text(
            x1 + 0.04,
            y_top,
            "bk-d" if target_syl == "bb" else f"{target_syl}-{target_branch}",
            va="center",
            ha="left",
            fontsize=max(5, 8),
        )
        ax.plot([x0, x1], [y_bottom, y_bottom], color=other_line_color, lw=1, zorder=10)
        ax.text(
            x1 + 0.04,
            y_bottom,
            "bk-i" if target_syl == "bb" else f"{target_syl}-{other_branch_label}",
            va="center",
            ha="left",
            fontsize=max(5, 8),
        )

    # --- Test baseline vs. last 3 days of training ---
    catch_stats = None    
    base_folders = [f for f in all_folders if f.lower().startswith("base")]
    train_folders = [f for f in all_folders if f.lower().startswith("train")]
    last3_train_folders = train_folders[-3:]

    if len(last3_train_folders) > 0:
        catch_stats = run_2x2_branch_test(
            labels_dict=labels_dict,
            folders_a=base_folders,
            folders_b=last3_train_folders,
            target_syl=target_syl,
            branch_syl=branch_syl,
            target_branch=target_branch,
            label_a="baseline",
            label_b="last3_train",
            dataset_name="baseline_vs_last3_train"
        )

    if catch_stats is not None:
        x_base = np.mean([
            folder_to_idx[f]
            for f in catch_stats["folders_a"]
            if f in folder_to_idx
        ])
        last3_x = [
            folder_to_idx[f]
            for f in catch_stats["folders_b"]
            if f in folder_to_idx
        ]
        if len(last3_x) > 0:
            x_train_last3_center = np.mean(last3_x)
            add_sig_bracket(
                ax,
                x_base,
                x_train_last3_center,
                y_ax=1.06,
                h=0.05,
                text=stars_from_p(catch_stats["p"]),
                lw=1.0
            )
            
    if catch_stats is not None:
        last3_x = [
            folder_to_idx[f]
            for f in catch_stats["folders_b"]
            if f in folder_to_idx
        ]
        if len(last3_x) >= 2:
            add_sig_bracket(
                ax,
                min(last3_x),
                max(last3_x),
                y_ax=1.015,
                h=0.025,
                text="",
                lw=1.0
            )

    post_folders = [f for f in all_folders if f.lower().startswith("post")]
    if post_folders and target_branch is not None:
        last_post = post_folders[-1]
        
        if target_branch in branch_syl:
            target_idx = branch_syl.index(target_branch)

            post_counts = count_branch_events_for_folders(
                labels_dict=labels_dict,
                folders=[last_post],
                target_syl=target_syl,
                branch_syl=branch_syl
            )

            x = int(post_counts[target_idx])
            y = int(post_counts.sum())

            if y > 0:
                prob = x / y

                print(f"last post day: {last_post}")
                print(
                    f"{x}/{y} = {prob * 100:.2f}%"
                )
            else:
                print(f"last post day: {last_post}")
                print(f"No {target_syl}→branch events found.")

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            safe_target = re.sub(r'[^A-Za-z0-9]+', '_', target_syl).strip('_')
            save_name = f"branchpoint_fullplot_{safe_target}.png"
        save_path = os.path.join(save_dir, save_name)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return (
        catch_matrix_full,
        all_folders,
        catch_stats
    )

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

    return day_data

def syllables_beforeafter_target(
    birds,
    target_syl_per_bird,
    position,
    target_repeat_per_bird=None,
    n_last_training_days=3,
    bird_name_mapping=None,
    bird_colors=None,
    figsize=(5, 6),
    y_label=None,
    save_dir=None,
    save_name=None,
    dpi=300,
    ):
    """
    Plots the mean number of syllables before or after the target for
    baseline and training for multiple experiments.

    Parameters
    ----------
    birds : dict
        Dictionary containing label sequences from multiple experiments over multiple days
    target_syl_per_bird : dict [str]
        The targeted repeat phrase for each experiment 
    position : str
        Number of syllables to count,
        only "before" or "after" are valid 
    target_repeat_per_bird : dict [int]
        Repeat number that was targeted in each experiment
    n_last_train_days : int
        Number of last training days to consider,
        default : 3        
    bird_name_mapping: dict
        Maps experiment names to more concise letters for the legend
    bird_colors : dict
        Maps a specific color to each experiment
    """

    if position not in ("before", "after"):
        raise ValueError("position must be 'before' or 'after'")

    if bird_name_mapping is None:
        bird_name_mapping = {}
    if bird_colors is None:
        bird_colors = {}
    bird_order = list(birds.keys())

    if y_label is None:
        if position == "before":
            y_label = r"n Syl. before Target"
        else:
            y_label = r"n Syl. after Target"

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

    bird_names = []
    display_names = []
    base_vals = []
    train_vals = []
    detailed = {}

    for bird in bird_order:

        if bird not in target_syl_per_bird:
            print(f"[skip] {bird}: no target_syl defined")
            continue

        if isinstance(target_repeat_per_bird, dict):
            target_repeat = target_repeat_per_bird.get(bird, None)
        else:
            target_repeat = target_repeat_per_bird

        b_vals, t_vals = extract_seq_counts(
            labels_dict=birds[bird],
            target_syl=target_syl_per_bird[bird],
            position=position,
            target_repeat=target_repeat,
            n_last_training_days=n_last_training_days
        )

        base_center = nanmean_or_nan(b_vals)
        train_center = nanmean_or_nan(t_vals)

        bird_names.append(bird)
        display_names.append(map_bird_name(bird, bird_name_mapping))
        base_vals.append(base_center)
        train_vals.append(train_center)

        detailed[bird] = {
            "target_syl": target_syl_per_bird[bird],
            "target_repeat": target_repeat,
            "baseline_raw": b_vals,
            "training_raw": t_vals,
            "baseline_mean": base_center,
            "training_mean": train_center,
            "n_baseline_values": len(b_vals),
            "n_training_values": len(t_vals),
        }

    if len(bird_names) == 0:
        print("[warn] No birds to plot.")
        return {"birds": [], "details": {}}

    base_arr = np.array(base_vals, dtype=float)
    train_arr = np.array(train_vals, dtype=float)

    valid_mask = np.isfinite(base_arr) & np.isfinite(train_arr)
    base_valid = base_arr[valid_mask]
    train_valid = train_arr[valid_mask]

    wilcoxon_stat = np.nan
    wilcoxon_p = np.nan
    wilcoxon_n = len(base_valid)

    if wilcoxon_n >= 1:
        diffs = train_valid - base_valid
        n_nonzero = np.sum(diffs != 0)

        if n_nonzero > 0:
            try:
                res = stats.wilcoxon(
                    base_valid,
                    train_valid,
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
        f"Wilcoxon signed-rank test ({position}, baseline vs training): "
        f"n={wilcoxon_n}, statistic={wilcoxon_stat}, p={wilcoxon_p}"
    )

    colors = [
        bird_colors.get(bird, wes_palette[i % len(wes_palette)])
        for i, bird in enumerate(bird_names)
    ]

    fig, ax = plt.subplots(figsize=figsize)

    x_base, x_train = setup_bs_t_summary_axis(ax)

    for i, bird in enumerate(bird_names):
        y_base = base_arr[i]
        y_train = train_arr[i]

        if np.isnan(y_base) and np.isnan(y_train):
            continue

        ax.plot(
            [x_base, x_train],
            [y_base, y_train],
            color=colors[i],
            lw=1,
            alpha=0.9,
            zorder=1,
            label=display_names[i],
        )

    group_base = nanmean_or_nan(base_arr)
    group_train = nanmean_or_nan(train_arr)

    group_base_sem = sem_of_valid(base_arr)
    group_train_sem = sem_of_valid(train_arr)

    if not (np.isnan(group_base) and np.isnan(group_train)):
        ax.errorbar(
            [x_base, x_train],
            [group_base, group_train],
            yerr=[group_base_sem, group_train_sem],
            color="black",
            lw=1.2,
            marker="o",
            markersize=4,
            capsize=3,
            elinewidth=1,
            alpha=1,
            zorder=5,
        )

    ax.set_ylabel(y_label)
    ax.set_ylim(0,)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y = 1.02
    h = 0.03

    ax.plot(
        [x_base, x_base, x_train, x_train],
        [y, y + h, y + h, y],
        transform=ax.get_xaxis_transform(),
        color="black",
        lw=1,
        clip_on=False,
    )

    p_text = stars_from_p(wilcoxon_p)

    ax.text(
        (x_base + x_train) / 2,
        y + h,
        p_text,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        clip_on=False,
    )

    handles, labels_leg = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels_leg,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            frameon=False,
            handlelength=0.6,
        )

    finalize_bs_t_summary_layout(fig)
    pos = ax.get_position()
    scale = 0.667
    ax.set_position([
        pos.x0 + pos.width * (1 - scale) / 2,
        pos.y0 + pos.height * (1 - scale) / 2,
        pos.width * scale,
        pos.height * scale,
    ])

    print(
        f"Baseline: {group_base:.2f} ± {group_base_sem:.2f}, "
        f"Training: {group_train:.2f} ± {group_train_sem:.2f}"
    )
    print(f"Wilcoxon test: n={wilcoxon_n}, statistic={wilcoxon_stat}, p={wilcoxon_p}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        if save_name is None:
            save_name = f"summary_syl_{position}_target.png"

        save_path = os.path.join(save_dir, save_name)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

def summary_target_repeat_per_bout(
    birds,
    target_syl_per_bird,
    figsize=(9, 4),
    n_last_training_days=3,
    bird_name_mapping=None,
    bird_colors=None,
    save_dir=None,
    save_name=None,
    dpi=300
    ):
    """
    Mean number of target repeat phrases in each bout in baseline vs. training for
    multiple experiments.

    Parameters
    ----------
    birds : dict
        Dictionary containing label sequences from multiple experiments over multiple days
    target_syl_per_bird : dict [str]
        The targeted repeat phrase for each experiment 
    n_last_train_days : int
        Number of last training days to consider,
        default : 3 
    bird_name_mapping: dict
        Maps experiment names to more concise letters for the legend
    bird_colors : dict
        Maps a specific color to each experiment       
    """

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

    bird_names = []
    bouts_baseline = []
    bouts_training = []
    repeats_baseline = []
    repeats_training = []
    repeats_baseline_sem = []
    repeats_training_sem = []

    for bird, labels_dict in birds.items():
        target_syl = target_syl_per_bird[bird]
        day_data = compute_bouts_and_targets_per_day(labels_dict, target_syl)

        baseline_days, training_days = split_baseline_training_days(day_data)
        baseline_days = sorted(baseline_days, key=lambda x: day_sort_key(x[0]))
        training_days = sorted(training_days, key=lambda x: day_sort_key(x[0]))

        if n_last_training_days is not None:
            training_days = training_days[-n_last_training_days:]

        base_bouts_arr = np.array([vals["n_bouts"] for _, vals in baseline_days], dtype=float)
        train_bouts_arr = np.array([vals["n_bouts"] for _, vals in training_days], dtype=float)
        base_bouts_mean = float(np.nansum(base_bouts_arr)) if len(base_bouts_arr) > 0 else np.nan
        train_bouts_mean = float(np.nansum(train_bouts_arr)) if len(train_bouts_arr) > 0 else np.nan

        base_rep_vals = flatten_repeats(baseline_days)
        train_rep_vals = flatten_repeats(training_days)
        base_rep_mean = nanmean_or_nan(base_rep_vals)
        train_rep_mean = nanmean_or_nan(train_rep_vals)
        bird_names.append(bird)
        bouts_baseline.append(base_bouts_mean)
        bouts_training.append(train_bouts_mean)
        repeats_baseline.append(base_rep_mean)
        repeats_training.append(train_rep_mean)
        repeats_baseline_sem.append(sem_of_valid(base_rep_vals))
        repeats_training_sem.append(sem_of_valid(train_rep_vals))

    bouts_baseline = np.array(bouts_baseline, dtype=float)
    bouts_training = np.array(bouts_training, dtype=float)
    repeats_baseline = np.array(repeats_baseline, dtype=float)
    repeats_training = np.array(repeats_training, dtype=float)
    repeats_baseline_sem = np.asarray(repeats_baseline_sem, dtype=float)
    repeats_training_sem = np.asarray(repeats_training_sem, dtype=float)

    display_bird_names = [map_bird_name(bird, bird_name_mapping) for bird in bird_names]
    colors = [
        bird_colors.get(bird, wes_palette[i % len(wes_palette)])
        for i, bird in enumerate(bird_names)
    ]

    print("\nSummary target repeats per song across birds:")
    for i, bird in enumerate(bird_names):
        print(
            f"{display_bird_names[i]} ({bird}): "
            f"repeats_per_song BS/T={repeats_baseline[i]:.2f}/{repeats_training[i]:.2f}"
        )

    # --- Wilcoxon signed-rank test: baseline vs training ---
    valid_mask = np.isfinite(repeats_baseline) & np.isfinite(repeats_training)
    base_valid = repeats_baseline[valid_mask]
    train_valid = repeats_training[valid_mask]

    wilcoxon_stat = np.nan
    wilcoxon_p = np.nan
    wilcoxon_n = len(base_valid)

    if wilcoxon_n >= 1:
        diffs = train_valid - base_valid
        n_nonzero = np.sum(diffs != 0)

        if n_nonzero > 0:
            try:
                res = stats.wilcoxon(
                    base_valid,
                    train_valid,
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
    x_base_pos, x_train_pos = setup_bs_t_summary_axis(ax)
    x_base = np.full(len(bird_names), x_base_pos)
    x_train = np.full(len(bird_names), x_train_pos)

    for i in range(len(bird_names)):
        if np.isnan(repeats_baseline[i]) and np.isnan(repeats_training[i]):
            continue

        ax.plot(
            [x_base[i], x_train[i]],
            [repeats_baseline[i], repeats_training[i]],
            color=colors[i],
            lw=1,
            alpha=0.9,
            zorder=1,
            label=display_bird_names[i],
        )
        for x, mean, sem in [
            (x_base[i], repeats_baseline[i], repeats_baseline_sem[i]),
            (x_train[i], repeats_training[i], repeats_training_sem[i]),
            ]:
            if not np.isfinite(mean):
                continue
            ax.scatter(
                x, mean,
                color=colors[i],
                marker="o",
                s=9,
                zorder=3,
            )
            if np.isfinite(sem):
                ax.errorbar(
                    x, mean,
                    yerr=sem,
                    fmt="none",
                    ecolor=colors[i],
                    elinewidth=1,
                    capsize=2,
                    zorder=2,
                )

    ax.set_ylabel("Target Phrases /\nBout")
    ax.set_ylim(0,)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # --- Group mean + SEM ---
    group_base = nanmean_or_nan(repeats_baseline)
    group_train = nanmean_or_nan(repeats_training)
    group_base_sem = sem_of_valid(repeats_baseline)
    group_train_sem = sem_of_valid(repeats_training)

    if not (np.isnan(group_base) and np.isnan(group_train)):
        ax.errorbar(
            [x_base_pos, x_train_pos],
            [group_base, group_train],
            yerr=[group_base_sem, group_train_sem],
            color="black",
            lw=1.2,
            marker="o",
            markersize=4,
            capsize=3,
            elinewidth=1,
            alpha=1,
            zorder=5
        )

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
        handlelength=0.6,
    )

    finalize_bs_t_summary_layout(fig)
    pos = ax.get_position()
    scale = 0.667

    ax.set_position([
        pos.x0 + pos.width * (1 - scale) / 2,
        pos.y0 + pos.height * (1 - scale) / 2,
        pos.width * scale,
        pos.height * scale
    ])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            save_name = "summary_targets_per_song.png"
        save_path = os.path.join(save_dir, save_name)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
        
    print(f"BS: {group_base:.2f}, +- {group_base_sem:.2f}")
    print(f"TR: {group_train:.2f}, +- {group_train_sem:.2f}")

def summary_targetrepeat_per_repeats(
    birds,
    target_syl_per_bird,
    target_repeat_per_bird,
    context_per_bird,
    figsize=(6, 6),
    show_bird_labels=False,
    n_last_training_days=3,
    bird_name_mapping=None,
    bird_colors=None,
    save_dir=None,
    save_name=None,
    dpi=300
    ):
    """
    Mean target repeat number appearance per target repeat phrase in baseline vs. training for 
    multiple experiments. 
    Branch point experiments cannot be considered.

    birds : dict
        Dictionary containing label sequences from multiple experiments over multiple days
    target_syl_per_bird : dict [str]
        The targeted repeat phrase for each experiment      
    target_repeat_per_bird : dict [int]
        Repeat number that was targeted in each experiment
    context_per_bird : dict [bool]
        For each target syllable whether it is context-dependent, True depicts context-dependency
    show_bird_labels : bool
        Optionally show experiment labels in the plot 
    n_last_train_days : int
        Number of last training days to consider,
        default : 3
    bird_name_mapping: dict
        Maps experiment names to more concise letters for the legend
    bird_colors : dict
        Maps a specific color to each experiment
    """

    if bird_name_mapping is None:
        bird_name_mapping = {}
    if bird_colors is None:
        bird_colors = {}

    bird_names = []
    baseline_vals = []
    training_vals = []
    baseline_sems = []
    training_sems = []
    baseline_total_repeats = []
    training_total_repeats = []
    baseline_target_repeats = []
    training_target_repeats = []

    for bird, labels_dict in birds.items():
        target_syl = target_syl_per_bird[bird]
        target_repeat = target_repeat_per_bird[bird]
        context = context_per_bird[bird]

        day_data = compute_day_raw_values(
            labels_dict=labels_dict,
            target_syl=target_syl,
            context=context
        )

        baseline_days, training_days = split_baseline_training_days(day_data)
        baseline_days = sorted(baseline_days, key=lambda x: day_sort_key(x[0]))
        training_days = sorted(training_days, key=lambda x: day_sort_key(x[0]))
        if n_last_training_days is not None:
            training_days = training_days[-n_last_training_days:]

        base_total = 0
        base_target = 0
        for day_name, vals in baseline_days:
            lengths = get_lengths_from_vals(vals)
            base_total += len(lengths)
            base_target += int(np.sum(lengths >= target_repeat))

        train_total = 0
        train_target = 0
        for day_name, vals in training_days:
            lengths = get_lengths_from_vals(vals)
            train_total += len(lengths)
            train_target += int(np.sum(lengths >= target_repeat))

        base_fraction = base_target / base_total if base_total > 0 else np.nan
        train_fraction = train_target / train_total if train_total > 0 else np.nan
        base_sem = (
            np.sqrt(base_fraction * (1.0 - base_fraction) / base_total)
            if base_total > 0 else np.nan
        )
        train_sem = (
            np.sqrt(train_fraction * (1.0 - train_fraction) / train_total)
            if train_total > 0 else np.nan
        )
        # Compare target-repeat proportions between baseline and training for each experiment
        if base_total > 0 and train_total > 0:
            table = np.array([
                [base_target, base_total - base_target],
                [train_target, train_total - train_target],
            ])
            if np.any(table.sum(axis=0) == 0):
                print(f"{bird}: No test possible (one category absent).")
            else:
                chi2, p_chi2, _, expected = stats.chi2_contingency(
                    table, correction=False
                )
                if np.all(expected >= 5):
                    print(f"{bird}: Chi-square test, statistic={chi2:.3f}, p={p_chi2:.4g}")
                else:
                    odds_ratio, p_fisher = stats.fisher_exact(table)
                    print(f"{bird}: Fisher's exact test, odds ratio={odds_ratio:.3f}, p={p_fisher:.4g}")
        bird_names.append(bird)
        baseline_vals.append(base_fraction)
        baseline_sems.append(base_sem)
        training_sems.append(train_sem)
        training_vals.append(train_fraction)
        baseline_total_repeats.append(base_total)
        training_total_repeats.append(train_total)
        baseline_target_repeats.append(base_target)
        training_target_repeats.append(train_target)

    baseline_vals = np.array(baseline_vals, dtype=float)
    training_vals = np.array(training_vals, dtype=float)
    baseline_sems = np.asarray(baseline_sems, dtype=float)
    training_sems = np.asarray(training_sems, dtype=float)
    baseline_total_repeats = np.array(baseline_total_repeats, dtype=float)
    training_total_repeats = np.array(training_total_repeats, dtype=float)
    baseline_target_repeats = np.array(baseline_target_repeats, dtype=float)
    training_target_repeats = np.array(training_target_repeats, dtype=float)

    display_bird_names = [map_bird_name(bird, bird_name_mapping) for bird in bird_names]

    print("\nSummary target repeat fraction across birds:")
    for i, bird in enumerate(bird_names):
        print(
            f"{display_bird_names[i]} ({bird}): "
            f"baseline={baseline_vals[i]:.4f} "
            f"({int(baseline_target_repeats[i])}/{int(baseline_total_repeats[i])}), "
            f"training={training_vals[i]:.4f} "
            f"({int(training_target_repeats[i])}/{int(training_total_repeats[i])})"
        )

    # --- Wilcoxon signed-rank test: baseline vs training ---
    valid_mask = np.isfinite(baseline_vals) & np.isfinite(training_vals)
    baseline_valid = baseline_vals[valid_mask]
    training_valid = training_vals[valid_mask]

    wilcoxon_stat = np.nan
    wilcoxon_p = np.nan
    wilcoxon_n = len(baseline_valid)

    if wilcoxon_n >= 1:
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

    x_base_pos, x_train_pos = setup_bs_t_summary_axis(ax)

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
            [x_base_pos, x_train_pos],
            [baseline_vals[i], training_vals[i]],
            color=colors[i],
            lw=1,
            alpha=0.9,
            zorder=1,
            label=display_bird_names[i]
        )
        for x, mean, sem in [
            (x_base_pos, baseline_vals[i], baseline_sems[i]),
            (x_train_pos, training_vals[i], training_sems[i]),
            ]:
            if not np.isfinite(mean):
                continue
            ax.scatter(x, mean, color=colors[i], s=9, zorder=3)
            if np.isfinite(sem):
                ax.errorbar(
                    x, mean,
                    yerr=sem,
                    fmt="none",
                    ecolor=colors[i],
                    elinewidth=1,
                    capsize=2,
                    zorder=2,
                ) 
        if show_bird_labels and not np.isnan(training_vals[i]):
            ax.text(
                x_train_pos + 0.03,
                training_vals[i],
                display_bird_names[i],
                va="center"
            )

    group_base = nanmean_or_nan(baseline_vals)
    group_train = nanmean_or_nan(training_vals)
    group_base_sem = sem_of_valid(baseline_vals)
    group_train_sem = sem_of_valid(training_vals)

    if (
        np.sum(np.isfinite(baseline_vals) & np.isfinite(training_vals)) > 1
        and not (np.isnan(group_base) and np.isnan(group_train))
        ):
        ax.errorbar(
            [x_base_pos, x_train_pos],
            [group_base, group_train],
            yerr=[group_base_sem, group_train_sem],
            color="black",
            lw=1.2,
            marker="o",
            markersize=4,
            capsize=3,
            elinewidth=1,
            alpha=1,
            zorder=5
        )

    ax.set_ylabel("Target Number /\nTarget Phrase")
    ax.set_ylim(0, )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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
    scale = 0.667
    ax.set_position([
        pos.x0 + pos.width * (1 - scale) / 2,
        pos.y0 + pos.height * (1 - scale) / 2,
        pos.width * scale,
        pos.height * scale
    ])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            save_name = "summary_targetrepeat_per_repeats.png"
        save_path = os.path.join(save_dir, save_name)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    print(f"BS: {group_base:.2f} ± {group_base_sem:.2f}")
    print(f"TR: {group_train:.2f} ± {group_train_sem:.2f}")

# ----- Run Main ----- #

if __name__ == "__main__":

    # S2A Fig
    plot_branchpoint_probabilities_fullplot(
        labels_dict=control_birds_catch["Control1_dj"],
        target_syl="d",
        branch_syl = ["j", "f"],
        target_branch="j",
        figsize=(1.9, 1.5),
        save_dir=save_dir,
        save_name="S2_Fig_A.svg",
        dpi=300
    )

    # S2B Fig
    plot_repeat_distributions_per_phase(
        labels_dict=control_birds_catch["Control2_C7"],
        target_syl="c+",
        color="#C8CA44",
        figsize=(1.9, 1.5),
        save_dir=save_dir,
        save_name="S2_Fig_B.svg",
        dpi=300
    )

    # S2C Fig left
    syllables_beforeafter_target(
        birds=control_birds_catch,
        target_syl_per_bird=target_syl_per_control_bird,
        target_repeat_per_bird=target_repeat_per_control_bird,
        position="after",
        bird_name_mapping=bird_name_mapping,
        bird_colors=control_colors,
        figsize=(1.525, 1.494),
        save_dir=save_dir,
        save_name="S2_Fig_C_1.svg",
        dpi=300,
    )

    # S2C Fig center
    summary_target_repeat_per_bout(
        birds=control_birds_catch,
        target_syl_per_bird=target_syl_per_control_bird,
        bird_name_mapping=bird_name_mapping,
        bird_colors=control_colors,
        figsize=(1.525, 1.494),
        save_dir=save_dir,
        save_name="S2_Fig_C_2.svg",
        dpi=300
    )

    # S2C Fig right
    summary_targetrepeat_per_repeats(
        birds= {bird : data
                for bird, data in control_birds_catch.items()
                if bird != "Control1_dj"},
        target_syl_per_bird=target_syl_per_control_bird,
        target_repeat_per_bird=target_repeat_per_control_bird,
        context_per_bird=context_per_control_bird,
        bird_name_mapping=bird_name_mapping,
        bird_colors=control_colors,
        figsize=(1.525, 1.494),
        save_dir=save_dir,
        save_name="S2_Fig_C_3.svg",
        dpi=300
    )
