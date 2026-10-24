"""
Code for results depicted in Supplementary Figure 4
    B) Transition probability plot over days
    C) Repeat distributions for baseline and training Bird2 B3+
    D) Repeat distributions for baseline and training Bird3 bF5+
    E) Repeat distributions for baseline and training Bird4 J5+
    F) Self-transition probability for meta repeat: within-chunk and between-chunk prob.

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
from statsmodels.stats.multitest import multipletests
from collections import defaultdict

from util.helper_fct import (
    load_birds,
    get_figure_dir,
    BIRD_COLORS as bird_colors
)

from util.helper_fct import (
    sem_of_valid, normality_from_values,
    get_repeat_lengths, day_sort_key, compute_branch_probs,
    count_branch_events_for_folders, add_sig_bracket,
    stars_from_p, plot_segments, run_2x2_branch_test
)

#----- Load data -----#

birds = load_birds()
birds_catch = load_birds(catch=True)

save_dir = get_figure_dir("FigureS4")

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

def plot_branchpoint_probabilities_fullplot(
    labels_dict,
    target_syl,
    branch_syl,
    target_branch=None,
    target_line_color="orange",
    other_line_color="#b3b3b3",
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

    tests_out = {}
    print("\n=== TWO-CHAR TRANSITION TESTS ===")
    print(f"target_syl={target_syl}, repeat_unit={repeat_unit}, context={context}")
    print(f"training_days_used={training_days}")
    print(
        f"matches: baseline={baseline_stats.get('n_matches', 0)}, "
        f"training={training_stats.get('n_matches', 0)}"
    )

    # Compare transition probabilities at each repeat position
    for key, label in [
        ("first_to_second", f"{first_char}->{second_char}"),
        ("second_to_first_next", f"{second_char}->{first_char}"),
    ]:
        base_pos_stats = baseline_stats.get(key, {})
        train_pos_stats = training_stats.get(key, {})
        positions = sorted(set(base_pos_stats) | set(train_pos_stats))
        one_test = {}

        for pos in positions:
            b = base_pos_stats.get(pos, {})
            t = train_pos_stats.get(pos, {})
            b_trans = int(b.get("n_transition", 0))
            b_other = int(b.get("n_other", 0))
            t_trans = int(t.get("n_transition", 0))
            t_other = int(t.get("n_other", 0))
            b_total = b_trans + b_other
            t_total = t_trans + t_other
            p_base = b_trans / b_total if b_total > 0 else np.nan
            p_train = t_trans / t_total if t_total > 0 else np.nan

            table = np.array([
                [b_trans, b_other],
                [t_trans, t_other],
            ], dtype=float)
            if (
                table.sum() == 0
                or np.any(table.sum(axis=0) == 0)
                or np.any(table.sum(axis=1) == 0)
            ):
                method = "n/a"
                stat_val = np.nan
                p_val = np.nan
            elif np.any(table < 5):
                method = "fisher"
                stat_val, p_val = stats.fisher_exact(
                    table,
                    alternative="two-sided",
                )
            else:
                method = "chi2"
                stat_val, p_val, _, _ = stats.chi2_contingency(
                    table,
                    correction=False,
                )
            one_test[pos] = {
                "baseline_transition": b_trans,
                "baseline_other": b_other,
                "training_transition": t_trans,
                "training_other": t_other,
                "p_baseline": p_base,
                "p_training": p_train,
                "method": method,
                "stat": float(stat_val) if np.isfinite(stat_val) else np.nan,
                "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
            }

        valid_positions = [
            pos for pos, result in one_test.items()
            if np.isfinite(result["p_value"])
        ]
        if valid_positions:
            p_values = [
                one_test[pos]["p_value"]
                for pos in valid_positions
            ]
            _, corrected_p, _, _ = multipletests(
                p_values,
                alpha=0.05,
                method="holm",
            )
            for pos, p_holm in zip(valid_positions, corrected_p):
                one_test[pos]["p_holm"] = float(p_holm)
        for pos in one_test:
            one_test[pos].setdefault("p_holm", np.nan)
        tests_out[key] = one_test

        print(f"\n{label}: Baseline vs Training per position")
        print(f"  Holm correction over positions for {label}:")
        for pos, result in sorted(one_test.items()):
            b_total = (
                result["baseline_transition"]
                + result["baseline_other"]
            )
            t_total = (
                result["training_transition"]
                + result["training_other"]
            )
            print(
                f"  pos {pos:>2}: "
                f"base={result['p_baseline']:.4f} "
                f"({result['baseline_transition']}/{b_total}), "
                f"train={result['p_training']:.4f} "
                f"({result['training_transition']}/{t_total}), "
                f"test={result['method']}, "
                f"stat={result['stat']:.4f}, "
                f"p={result['p_value']:.4g}, "
                f"p_holm={result['p_holm']:.4g}"
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

    return {
        "repeat_unit": repeat_unit,
        "baseline": baseline_stats,
        "training": training_stats,
        "tests": tests_out,
        "delta_summary": delta_summary,
        "training_days_used": training_days,
        "fig": fig,
    }

#----- Run Main -----#

if __name__ == "__main__":

    # Fig S2 B
    plot_branchpoint_probabilities_fullplot(
        labels_dict=birds_catch["Bird6_ah"],
        target_syl="a",
        branch_syl = ["h", "t"],
        target_branch="h",
        target_line_color=bird_colors["Bird6_ah"],
        figsize=(1.9, 1.5),
        save_dir=save_dir,
        save_name="S4_Fig_B.svg",
        dpi=300
    )

    # Fig S2 C
    plot_repeat_distributions_per_phase(
        labels_dict=birds_catch["Bird2_B3"],
        target_syl="b+",
        color=bird_colors["Bird2_B3"],
        figsize=(1.9, 1.5),
        save_dir=save_dir,
        save_name="S4_Fig_C.svg",
        dpi=300
    )

    # Fig S2 D
    plot_repeat_distributions_per_phase(
        labels_dict=birds_catch["Bird3_bF5"],
        target_syl="bf+",
        context=True,
        color=bird_colors["Bird3_bF5"],
        figsize=(1.9, 1.5),
        save_dir=save_dir,
        save_name="S4_Fig_D.svg",
        dpi=300
    )

    # Fig S2 E
    # I am using 'jm' here as the same syllable can be labelled 'j' or 'm' in the original sequence
    plot_repeat_distributions_per_phase(
        labels_dict=birds_catch["Bird4_J5"],
        target_syl="[jm]+",
        color=bird_colors["Bird4_J5"],
        figsize=(1.9, 1.5),
        save_dir=save_dir,
        save_name="S4_Fig_E.svg",
        dpi=300
    )

    # Fig S2 F
    metarepeat_transitions(
        labels_dict=birds_catch["Bird4_HF4"],
        target_syl="(?:hf)+",
        context=False,
        n_last_training_days=3,
        figsize=(1.9, 1.5),
        save_dir=save_dir,
        save_name="S4_Fig_F.svg",
        dpi=300,
        train_color=bird_colors["Bird4_HF4"]
    )
