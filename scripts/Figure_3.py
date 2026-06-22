"""
Code for results depicted in Figure 3
    B) Repeat distributions for baseline and training
    C) Convergence probability for the repeat syllable
    D) Transition probability at a branch point
    E) Transition diagram for baseline song
    E) Difference matrix baseline vs. training

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
from matplotlib.colors import LinearSegmentedColormap

from util.helper_fct import (
    load_birds,
    get_figure_dir
)

from util.helper_fct import (
    sem_of_valid, normality_from_values,
    get_repeat_lengths, day_sort_key,
    finalize_bs_t_summary_layout,
    stars_from_p, phase_from_day, add_sig_bracket,
    compute_transition_matrix_and_labels_from_bouts,
    print_transition_probabilities_after_thresholds,
    plot_transition_diagram
)

#----- Load data -----#

birds = load_birds()
birds_catch = load_birds(catch=True)

save_dir = get_figure_dir("Figure3")

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

def convergence_probability_scatter(
    labels_dict,
    repeats,
    last_n_train_days = 3,
    repeat_labels=None,
    highlight_repeat = None,
    colors=None,
    figsize=(3.5, 2.5),
    save_dir=None,
    save_name=None,
    dpi=300
):
    """
    Plot convergence probability for a given repeat for baseline vs. last n days of training

    Parameters
    ----------
    labels_dict : dict
        Dictionary containing label sequences over one or multiple days
    repeats : list [str]
        Repeat patterns to test, given as regular expressions, e.g. ["kb+", "ib+", "fb+"].
        The order also determines the plotting order.
    last_n_train_days : int 
        Number of last training days to include,
        default : 3, if None, all training days are used.
    repeat_labels : list [str]
        Optional display labels for the repeat patterns,
        if None, "repeats" are used as labels
    highlight_repeat : str
        Optional, the target repeat, will be highlighted in a different color
    colors : tuple
        Optional custom colors

    Returns
    -------
    dict
        Dictionary containing:
        - "conv_probs": convergence probabilities for each repeat and phase
        - "sems": binomial SEMs for each repeat and phase
        - "per_phase_repeat": raw repeat counts per phase
        - "per_phase_total": total matched repeat-context counts per phase
        - "repeat_vs_other_test": proportion test results for each repeat vs. all others.

    """

    per_phase_repeat = {r: {"Baseline": 0, "Training": 0} for r in repeats}
    per_phase_total = {"Baseline": 0, "Training": 0}

    all_train_days = sorted(
        {
            os.path.basename(os.path.dirname(p))
            for p, labels in labels_dict.items()
            if labels
            and os.path.basename(os.path.dirname(p)).lower().startswith("train")
        },
        key=day_sort_key
    )

    if last_n_train_days is not None:
        train_days_used = set(all_train_days[-last_n_train_days:])
    else:
        train_days_used = set(all_train_days)

    # search pattern before each target
    for file_path, labels in labels_dict.items():
        if not labels or not isinstance(labels, str):
            continue

        day_folder = os.path.basename(os.path.dirname(file_path))
        phase = phase_from_day(day_folder)

        if phase not in ("Baseline", "Training"):
            continue
        if phase == "Training" and day_folder not in train_days_used:
            continue
        for i in range(1, len(labels)):
            if labels[i] != "b":
                continue
            found = False
            for r in sorted(repeats, key=len, reverse=True):
                pat = re.compile(r)
                for m in pat.finditer(labels[:i]):
                    if m.end() == i:
                        per_phase_repeat[r][phase] += 1
                        per_phase_total[phase] += 1
                        found = True
                        break
                if found:
                    break
                
    conv_probs = {r: {} for r in repeats}
    sems = {r: {} for r in repeats}

    for repeat in repeats:
        for phase in ("Baseline", "Training"):
            total = per_phase_total[phase]
            count = per_phase_repeat[repeat][phase]

            if total > 0:
                p = count / total
                conv_probs[repeat][phase] = p
                sems[repeat][phase] = np.sqrt(p * (1 - p) / total)
            else:
                conv_probs[repeat][phase] = np.nan
                sems[repeat][phase] = np.nan

    proportion_test_results = {}

    print("\nRepeat-wise proportion tests: each repeat vs all others (Baseline vs Training)")

    for repeat in repeats:
        repeat_base = int(per_phase_repeat[repeat]["Baseline"])
        repeat_train = int(per_phase_repeat[repeat]["Training"])

        total_base = int(per_phase_total["Baseline"])
        total_train = int(per_phase_total["Training"])

        other_base = total_base - repeat_base
        other_train = total_train - repeat_train

        table = np.array([
            [repeat_base, other_base],
            [repeat_train, other_train]
        ], dtype=int)

        if total_base > 0 and total_train > 0 and np.all(table >= 0):
            chi2_tmp, _, _, expected = stats.chi2_contingency(table, correction=False)
            use_fisher = np.any(expected < 5)

            if use_fisher:
                stat_val, p_val = stats.fisher_exact(table, alternative="two-sided")
                test_name = "Fisher's Exact Test"
                stat_name = "oddsratio"
                dof = None
            else:
                stat_val, p_val, dof, expected = stats.chi2_contingency(table, correction=False)
                test_name = "Chi² contingency test"
                stat_name = "chi2"

            repeat_prop_base = (repeat_base / total_base) if total_base > 0 else np.nan
            repeat_prop_train = (repeat_train / total_train) if total_train > 0 else np.nan

            print(f"\n{repeat} vs all other repeats: Baseline vs Training")
            print("Contingency table [[repeat_base, other_base], [repeat_train, other_train]]:")
            print(table)
            print(f"Baseline: {repeat}={repeat_base}/{total_base} ({repeat_prop_base:.2f})")
            print(f"Training: {repeat}={repeat_train}/{total_train} ({repeat_prop_train:.2f})")
            print("Expected frequencies:")
            print(np.round(expected, 3))
            print(f"Chosen test: {test_name}")

            if dof is None:
                print(f"{stat_name}={stat_val:.4f}, p={p_val:.6g}, signif={stars_from_p(p_val)}")
            else:
                print(f"{stat_name}={stat_val:.4f}, dof={dof}, p={p_val:.6g}, signif={stars_from_p(p_val)}")

            proportion_test_results[repeat] = {
                "repeat_tested": repeat,
                "table": table,
                "expected": expected,
                "test_name": test_name,
                "stat_name": stat_name,
                "stat_value": float(stat_val),
                "dof": dof,
                "p_value": float(p_val),
                "baseline_count": repeat_base,
                "training_count": repeat_train,
                "other_baseline_count": other_base,
                "other_training_count": other_train,
                "baseline_percent": repeat_prop_base,
                "training_percent": repeat_prop_train,
            }
        else:
            print(f"\n{repeat} vs all other repeats: test not possible.")
            proportion_test_results[repeat] = None

    fig, ax = plt.subplots(figsize=figsize)
    x_base, x_train = 0, 1
    xs = [x_base, x_train]

    highlight_colors = ("#6e2222","#942D2D") 
    gray_colors = ("#030505ff","#999b9bff", "#c6cdcdff","#156666ff")

    if colors is None:
        base_colors = []
        train_colors = []
        line_colors = []
        for i,r in enumerate(repeats):
            if r == highlight_repeat:
                base_colors.append(highlight_colors[1])
                train_colors.append(highlight_colors[1])
                line_colors.append(highlight_colors[1])
            else:
                base_colors.append(gray_colors[i])
                train_colors.append(gray_colors[i])
                line_colors.append(gray_colors[i])

    else:
        base_colors, train_colors = colors
        line_colors = train_colors

    if repeat_labels is None:
        repeat_labels = [str(r) for r in repeats]

    for i, repeat in enumerate(repeats):
        y_base = conv_probs[repeat]["Baseline"]
        y_train = conv_probs[repeat]["Training"]
        sem_base = sems[repeat]["Baseline"]
        sem_train = sems[repeat]["Training"]
        line_color = line_colors[i % len(line_colors)]

        ax.plot(xs, [y_base, y_train], color=line_color, alpha=1, zorder=2)
        ax.errorbar(
            xs, [y_base, y_train],
            yerr=[sem_base, sem_train],
            fmt="none",
            ecolor=line_color,
            linewidth=1,
            capsize=3,
            zorder=3
        )

    ax.set_xticks([x_base, x_train])
    ax.set_xticklabels(["B", "T"])
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylabel("Convergence Prob.")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    add_sig_bracket(
        ax,
        x_base,
        x_train,
        text=stars_from_p(proportion_test_results[highlight_repeat]["p_value"]),
        y_ax=1.02,
        h=0.03,
        lw=1
    )

    finalize_bs_t_summary_layout(fig)
    pos = ax.get_position()
    scale = 0.654
    ax.set_position([
        pos.x0 + pos.width * (1 -  scale) / 2,
        pos.y0 + pos.height * (1 - scale) / 2,
        pos.width * scale,
        pos.height * scale
    ])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            save_name = "convergence_probability.png"
        save_path = os.path.join(save_dir, save_name)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()

    return {
        "conv_probs": conv_probs,
        "sems": sems,
        "per_phase_repeat": per_phase_repeat,
        "per_phase_total": per_phase_total,
        "repeat_vs_other_test": proportion_test_results,
    }

def plot_branchpoint_probabilities_scatter(
    labels_dict,
    target_syl,
    branch_syls,
    n_last_training_days=3,
    figsize=(3, 2),
    save_dir=None,
    save_name=None,
    dpi=300
    ):
    """
    Plot transition probability at a given branch point for baseline vs. training

    Parameters
    ----------
    labels_dict : dict
        Dictionary containing label sequences over one or multiple days
    target_syl : str
        The branch point syllable
    branch_syls : list [str]
        (Multiple) possible branches
    last_n_train_days : int 
        Number of last training days to include,
        default : 3, if None, all training days are used.

    Returns
    -------
    Mean and SEM for each branch, statistics
    """

    color_palette = [
        "#961919ff", "#bca371ff", "#b3b3b3", "#4d80ffff", "#26d9e4ff", "#1f7a1f", # kb2 #f08080ff als 1st color
        "#942D2D", "#F28E2B", "#1F7426", "#7BE966", "#ff69b4",
        "pink", "orange", "plum"
    ]
    color_map = {b: color_palette[i % len(color_palette)] for i, b in enumerate(branch_syls)}

    # Event-pooled means, sem
    means = {b: {"Baseline": np.nan, "Training": np.nan} for b in branch_syls}
    sems = {b: {"Baseline": np.nan, "Training": np.nan} for b in branch_syls}

    pooled_counts = {
        b: {
            "Baseline_target_to_b": 0,
            "Baseline_target_total": 0,
            "Training_target_to_b": 0,
            "Training_target_total": 0,
        }
        for b in branch_syls
    }

    target_pattern = re.escape(target_syl) + r"(.)"

    training_days = sorted(
        {
            os.path.basename(os.path.dirname(path))
            for path in labels_dict.keys()
            if os.path.basename(os.path.dirname(path)).lower().startswith("train")
        },
        key=day_sort_key,
    )

    if n_last_training_days is not None and n_last_training_days != 0:
        training_days = training_days[-int(n_last_training_days):]

    training_day_set = set(training_days)

    for file_path, labels in labels_dict.items():
        folder = os.path.basename(os.path.dirname(file_path))

        if folder.lower().startswith("base"):
            phase = "Baseline"
        elif folder.lower().startswith("train"):
            if folder not in training_day_set:
                continue
            phase = "Training"
        else:
            continue

        seq = labels
        if not isinstance(seq, str) or not seq:
            continue

        all_branches = re.findall(target_pattern, seq)
        if not all_branches:
            continue

        n_target = len(all_branches)

        for b in branch_syls:
            n_branch = all_branches.count(b)

            if phase == "Baseline":
                pooled_counts[b]["Baseline_target_to_b"] += n_branch
                pooled_counts[b]["Baseline_target_total"] += n_target
            else:
                pooled_counts[b]["Training_target_to_b"] += n_branch
                pooled_counts[b]["Training_target_total"] += n_target

    for b in branch_syls:
        for phase in ("Baseline", "Training"):
            n_branch = pooled_counts[b][f"{phase}_target_to_b"]
            n_total = pooled_counts[b][f"{phase}_target_total"]

            if n_total > 0:
                p = n_branch / n_total
                means[b][phase] = p 

                sems[b][phase] = np.sqrt(p * (1 - p) / n_total) 

    stats_results = None
    if len(branch_syls) == 2:
        b1, b2 = branch_syls

        b1_base = int(pooled_counts[b1]["Baseline_target_to_b"])
        b1_train = int(pooled_counts[b1]["Training_target_to_b"])

        b2_base = int(pooled_counts[b2]["Baseline_target_to_b"])
        b2_train = int(pooled_counts[b2]["Training_target_to_b"])

        table = np.array([
            [b1_base, b2_base],
            [b1_train, b2_train]
        ], dtype=int)

        if np.any(table.sum(axis=0) == 0) or np.any(table.sum(axis=1) == 0):
            print(f"\n[{target_syl}] no test possible.")
        else:
            _, _, _, expected = stats.chi2_contingency(table, correction=False)
            use_fisher = np.any(expected < 5)

            if use_fisher:
                stat_val, p_val = stats.fisher_exact(table, alternative="two-sided")
                test_name = "Fisher's Exact Test"
                stat_name = "oddsratio"
                dof = None
            else:
                stat_val, p_val, dof, expected = stats.chi2_contingency(table, correction=False)
                test_name = "Chi² contingency test"
                stat_name = "chi2"

            p_b1_base = b1_base / (b1_base + b2_base) if (b1_base + b2_base) > 0 else np.nan
            p_b1_train = b1_train / (b1_train + b2_train) if (b1_train + b2_train) > 0 else np.nan

            p_b2_base = b2_base / (b1_base + b2_base) if (b1_base + b2_base) > 0 else np.nan
            p_b2_train = b2_train / (b1_train + b2_train) if (b1_train + b2_train) > 0 else np.nan

            print(f"\n[{target_syl}] Comparison branches: {b1} vs {b2}")
            print("Contingency table [[base_b1, base_b2], [train_b1, train_b2]]:")
            print(table)

            print(
                f"Baseline: {b1}={b1_base} ({p_b1_base:.2f}), "
                f"{b2}={b2_base} ({p_b2_base:.2f})"
            )

            print(
                f"Training: {b1}={b1_train} ({p_b1_train:.2f}), "
                f"{b2}={b2_train} ({p_b2_train:.2f})"
            )

            print("Expected frequencies:")
            print(np.round(expected, 3))

            if dof is None:
                print(
                    f"Chosen test: {test_name} | {stat_name}={stat_val:.4f}, "
                    f"p={p_val:.6g}, signif={stars_from_p(p_val)}"
                )
            else:
                print(
                    f"Chosen test: {test_name} | {stat_name}={stat_val:.4f}, dof={dof}, "
                    f"p={p_val:.6g}, signif={stars_from_p(p_val)}"
            )

            stats_results = {
                "branches": (b1, b2),
                "table": table,
                "expected": expected,
                "test_name": test_name,
                "stat_name": stat_name,
                "stat_value": float(stat_val),
                "dof": dof,
                "p_value": float(p_val),
            }
    else:
        print(f"\n[{target_syl}] len(branch_syls) != 2.")
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    xs = [0, 1]

    for i, b in enumerate(branch_syls):
        yvals = [
            means[b]["Baseline"] if not np.isnan(means[b]["Baseline"]) else np.nan,
            means[b]["Training"] if not np.isnan(means[b]["Training"]) else np.nan
        ]
        semvals = [
            sems[b]["Baseline"] if not np.isnan(sems[b]["Baseline"]) else 0.0,
            sems[b]["Training"] if not np.isnan(sems[b]["Training"]) else 0.0
        ]
        ax.plot(xs, yvals, color=color_map[b], zorder=2)
        ax.errorbar(
            xs, yvals,
            yerr=semvals,
            fmt="none",
            ecolor=color_map[b],
            elinewidth=1.5,
            capsize=3,
            zorder=3
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(["B", "T"])
    ax.set_ylabel("Transition Prob.")
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Signifikanzklammern
    if stats_results is not None:
        add_sig_bracket(
            ax,
            0,
            1,
            text=stars_from_p(stats_results["p_value"]),
            y_ax=1.02,
            h=0.03,
            lw=1
        )

    finalize_bs_t_summary_layout(fig)
    pos = ax.get_position()
    scale = 0.66
    ax.set_position([
        pos.x0 + pos.width * (1 - 0.01 - scale) / 2,
        pos.y0 + pos.height * (1 - scale) / 2,
        pos.width * scale,
        pos.height * scale
    ])


    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            save_name = f"branchpoint_scatter_{target_syl}.png"
        save_path = os.path.join(save_dir, save_name)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()
    return {
        "means": means,
        "sems": sems,
        "stats": stats_results,
    }

def create_transition_diagram_from_labels_dict(
    labels_dict,
    title="Transition Diagram",
    save_path=None,
    node_threshold=0,
    edge_threshold=1,
    figsize=(10, 10),
    folder_prefix=None,
    label_merge_map=None,
    repeat_collapse_map=None,
    combine_bk_chunks=False,
):
    """
    Creates a transition diagram from a sequence of labels.
    Transitions are depicted with numbers and arrows between nodes. 
    Transition probabilities are calculated without excluded nodes/ edges.
    Prints individual transition probabilities after exclusion.

    Parameters
    ----------
    labels_dict : dict
        Dictionary containing label sequences over one or multiple days
    title : str
        Title of the final plot
    node_threshold : int
        Probability threshold of appearance for a node to be shown in the plot
    edge_threshold : int 
        Probability threshold of a transition to be shown in the plot
    folder_prefix : str
        Prefix of the files/ folders to be plotted, e.g. "base"
    label_merge_map : dict
        Map of labels that will be merged in the final plot, 
        transition probabilities within merged labels will be excluded
    repeat_collapse_map : dict
        Map labels whose consecutive repeats should be collapsed into one state,
        e.g. {"b": "B"} turns b b b d b b into B d B.
        Special case for Bird 1, mapping syllables 'bb' to 'bk'
    """

    if folder_prefix is not None:
        labels_dict = {
            path: seq
            for path, seq in labels_dict.items()
            if os.path.basename(os.path.dirname(path)).lower().startswith(folder_prefix.lower())
        }

    bout_list = [
        seq
        for seq in labels_dict.values()
        if isinstance(seq, str) and len(seq) > 0
    ]

    all_bouts = "".join(bout_list)

    print(f"  Total sequences: {len(bout_list)}")
    print(f"  Total bout length: {len(all_bouts)}")

    transition_matrix, unique_labels = compute_transition_matrix_and_labels_from_bouts(
        bout_list,
        label_merge_map=label_merge_map,
        repeat_collapse_map=repeat_collapse_map,
        combine_bk_chunks=combine_bk_chunks,
    )
    print_transition_probabilities_after_thresholds(
        transition_matrix,
        unique_labels,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )

    print(f"  Unique labels: {unique_labels}")
    print(f"  Transition matrix shape: {transition_matrix.shape}")

    fig, ax = plot_transition_diagram(
        transition_matrix,
        unique_labels,
        title=title,
        save_path=save_path,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        figsize=figsize,
    )

    plt.show()

    return {
        "transition_matrix": transition_matrix,
        "unique_labels": unique_labels,
        "bout_list": bout_list,
        "all_bouts": all_bouts
    }

def plot_transition_matrix_comparison(
    labels_dict,
    chunks=None,
    figsize=(5, 5),
    save_dir=None,
    save_name=None,
    dpi=300
):
    """
    Plots a transition-probability difference matrix of training - baseline

    Rows are "from" syllables/chunks, columns are "to" syllables/chunks.
    Values are differences in row-wise transition probabilities in percentage points.
    
    Parameters
    ----------
    labels_dict : dict
        Dictionary containing label sequences over one or multiple days
    chunks : list [str]
        Optional, two or more syllables that will be chunked to one syllables,
        transition probabilities within this chunk will be excluded from calculations,
        e.g. [("GHA", "g"), ("GHA", "h"), ("GHA", "a")] -> "g", "h" and "a" become "GHA"

    Returns
    -------
    Dict:
        "symbols" :
            Syllables/ Chunks shown in the matrix
        "counts_baseline" :
            absolute transition counts in baseline, from x to y
        "probs_baseline" : 
            transition probabilities in baseline in percent
        "counts_training" :
            absolute transition counts in training, from x to y
        "probs_training" : 
            transition probabilities in training in percent
        "diff_training_minus_baseline" :
            difference percentage points plotted in the matrix  
    """

    colorbar_kwargs = dict(
        fraction=0.035,
        pad=0.03,
        shrink=0.8,
        aspect=25
    )

    # --- chunk handling ---
    chunk_rules = []
    if chunks is not None:
        if isinstance(chunks, dict):
            chunk_items = list(chunks.items())
        elif isinstance(chunks, (list, tuple)):
            chunk_items = list(chunks)
        else:
            raise ValueError("chunks must be None, dict, or list/tuple of (token, sequence)")
        for item in chunk_items:
            token, seq = str(item[0]), str(item[1])
            if not token or not seq:
                raise ValueError("Chunk token and sequence must be non-empty")
            chunk_rules.append((token, seq))

        chunk_rules.sort(key=lambda x: len(x[1]), reverse=True)

    def tokenize_sequence(seq):
        if not chunk_rules:
            return list(seq)
        tokens = []
        i = 0
        while i < len(seq):
            matched = False
            for token, pattern in chunk_rules:
                if seq.startswith(pattern, i):
                    tokens.append(token)
                    i += len(pattern)
                    matched = True
                    break
            if not matched:
                tokens.append(seq[i])
                i += 1

        # collapse repeated chunk tokens, e.g. dc dc dc -> DC
        chunk_token_set = {token for token, _ in chunk_rules}

        if len(tokens) == 0:
            return tokens
        collapsed = [tokens[0]]
        for token in tokens[1:]:
            if token in chunk_token_set and token == collapsed[-1]:
                continue
            collapsed.append(token)

        return collapsed

    # --- split baseline / training ---
    base_token_seqs = []
    train_token_seqs = []

    for file_path, labels in labels_dict.items():
        if not isinstance(labels, str) or not labels:
            continue

        day = os.path.basename(os.path.dirname(file_path)).lower()
        tokenized = tokenize_sequence(labels)

        if day.startswith("base"):
            base_token_seqs.append(tokenized)

        elif day.startswith("train"):
            train_token_seqs.append(tokenized)

    if len(base_token_seqs) == 0:
        raise ValueError("No baseline sequences found")

    if len(train_token_seqs) == 0:
        raise ValueError("No training sequences found")

    # --- symbols ---
    symbols = sorted({
        token
        for seq in base_token_seqs + train_token_seqs
        for token in seq
    })

    idx = {symbol: i for i, symbol in enumerate(symbols)}

    # --- count transitions ---
    counts_base = np.zeros((len(symbols), len(symbols)), dtype=int)
    counts_train = np.zeros((len(symbols), len(symbols)), dtype=int)

    for seq in base_token_seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            counts_base[idx[a], idx[b]] += 1

    for seq in train_token_seqs:
        for a, b in zip(seq[:-1], seq[1:]):
            counts_train[idx[a], idx[b]] += 1

    # --- row-wise probabilities in percent ---
    row_sum_base = counts_base.sum(axis=1, keepdims=True)
    row_sum_train = counts_train.sum(axis=1, keepdims=True)

    probs_base = np.zeros_like(counts_base, dtype=float)
    probs_train = np.zeros_like(counts_train, dtype=float)

    np.divide(
        counts_base,
        row_sum_base,
        out=probs_base,
        where=row_sum_base != 0,
    )

    np.divide(
        counts_train,
        row_sum_train,
        out=probs_train,
        where=row_sum_train != 0,
    )

    probs_base *= 100
    probs_train *= 100

    diff = probs_train - probs_base

    # --- colormap for difference matrix ---
    rdbu = plt.get_cmap("RdBu_r")

    left = [rdbu(i / 255) for i in range(20, 128)]
    right = [rdbu(i / 255) for i in range(128, 236)]

    cmap_diff = LinearSegmentedColormap.from_list(
        "light_rdbu",
        left + [(1, 1, 1)] + right,
        N=256
    )

    vmax = max(1.0, np.max(np.abs(diff)))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        diff,
        cmap=cmap_diff,
        vmin=-vmax,
        vmax=vmax
    )

    ax.set_title("Training − Baseline")
    ax.set_xlabel("To Syllable")
    ax.set_ylabel("From Syllable")

    ax.set_xticks(np.arange(len(symbols)))
    ax.set_yticks(np.arange(len(symbols)))
    ax.set_xticklabels(symbols)
    ax.set_yticklabels(symbols)

    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            val = diff[i, j]

            ax.text(
                j,
                i,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if abs(val) > vmax * 0.5 else "black"
            )

    fig.colorbar(im, ax=ax, **colorbar_kwargs)
    fig.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        if save_name is None:
            save_name = "transition_matrix_comparison.svg"

        save_path = os.path.join(save_dir, save_name)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()

    return {
        "symbols": symbols,
        "counts_baseline": counts_base,
        "probs_baseline": probs_base,
        "counts_training": counts_train,
        "probs_training": probs_train,
        "diff_training_minus_baseline": diff,
    }

#----- Run Main -----#

if __name__ == "__main__":

    # Fig 3B "kB"
    plot_repeat_distributions_per_phase(
        labels_dict=birds_catch["Bird3_kB4"],
        target_syl="kb+",
        ylim = 0.6,
        context=True,
        color="#942d2dff",
        figsize=(1.9, 1.5),
        save_dir=get_figure_dir("Figure3"),
        save_name="Fig3_B_kB.svg",
        dpi=300
    )

    # Fig 3B "fB"
    plot_repeat_distributions_per_phase(
        labels_dict=birds_catch["Bird3_kB4"],
        target_syl="fb+",
        ylim = 0.6,
        context=True,
        color="#b1ababff",
        figsize=(1.9, 1.5),
        save_dir=get_figure_dir("Figure3"),
        save_name="Fig3_B_fB.svg",
        dpi=300
    )

    # Fig 3B "iB"
    plot_repeat_distributions_per_phase(
        labels_dict=birds_catch["Bird3_kB4"],
        target_syl="ib+",
        ylim = 0.6,
        context=True,
        color="#726f6fff",
        figsize=(1.9, 1.5),
        save_dir=get_figure_dir("Figure3"),
        save_name="Fig3_B_iB.svg",
        dpi=300
    )

    # Fig 3 C
    convergence_probability_scatter(
        labels_dict = birds_catch["Bird3_kB4"],
        repeats = ['kb+', 'ib+', 'fb+'],
        highlight_repeat='kb+',
        figsize=(1.9, 1.5),
        save_dir=get_figure_dir("Figure3"),
        save_name="Fig3_C.svg",
        dpi=300
    )

    # Fig 3 D
    plot_branchpoint_probabilities_scatter(
        labels_dict = birds_catch["Bird3_kB4"],
        target_syl="k", 
        branch_syls=["b","c"],
        figsize=(1.9, 1.5),
        save_dir=get_figure_dir("Figure3"),
        save_name="Fig3_D.svg",
        dpi=300
    )

    # Fig 3 E Baseline
    create_transition_diagram_from_labels_dict(
        labels_dict=birds["Bird3_kB4"],
        title="Baseline",
        folder_prefix="base",
        label_merge_map={"g": "gha", "h": "gha", "a": "gha"},  # g/h/a → gha
        save_path=str(get_figure_dir("Figure3")/"Fig3_E_Baseline.svg"),
        node_threshold=1,
        edge_threshold=5,
        combine_bk_chunks=False,
        figsize=(5,5)
    )

    # Fig 3 E Comparison Matrix
    plot_transition_matrix_comparison(
        birds_catch["Bird3_kB4"],
        chunks=[("GHA", "g"), ("GHA", "h"), ("GHA", "a")],
        figsize=(4.5, 3),
        save_dir=get_figure_dir("Figure3"),
        save_name="Fig3_E_ComparisonMatrix.svg",
        dpi=300
    )
