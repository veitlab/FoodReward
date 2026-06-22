"""
Code for results depicted in Figure 1
    C) Transition diagram 
    D) Transition probability plot over days

Functions for the code repository accompanying:
Birdsong Modification with Food Reward

Author: Franziska Heubach
Year:2026
"""

#----- Imports -----#

import os
import re
import matplotlib.pyplot as plt
import numpy as np

from util.helper_fct import (
    load_birds,
    get_figure_dir
)
from util.helper_fct import (
    compute_transition_matrix_and_labels_from_bouts, 
    print_transition_probabilities_after_thresholds, 
    plot_transition_diagram, compute_branch_probs, 
    plot_segments, run_2x2_branch_test, add_sig_bracket, 
    stars_from_p, count_branch_events_for_folders)

#----- Load data -----#

birds = load_birds()
birds_catch = load_birds(catch=True)

save_dir = get_figure_dir("Figure1")

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

    plt.show()

    return (
        catch_matrix_full,
        all_folders,
        catch_stats
    )

#----- Run Main -----#

if __name__ == "__main__":
    
    # Fig 1C Baseline
    create_transition_diagram_from_labels_dict(
        labels_dict=birds["Bird1_bkd"],
        title="Baseline",
        folder_prefix="base",
        save_path=str(get_figure_dir("Figure1")/"Fig1_C_Baseline.svg"),
        node_threshold=1,
        edge_threshold=4,
        combine_bk_chunks=True,
        figsize=(5,5)
    )

    # Fig 1C Training
    create_transition_diagram_from_labels_dict(
        labels_dict=birds_catch["Bird1_bkd"],
        title="Final Training",
        folder_prefix="train_241007",
        save_path=str(get_figure_dir("Figure1")/"Fig1_C_Training.svg"),
        node_threshold=1,
        edge_threshold=4,
        combine_bk_chunks=True,
        figsize=(5,5)
    )

    # Fig 1D
    # I am using "bb" here, as "b" and "k" are both labeled "b" in the original sequence
    plot_branchpoint_probabilities_fullplot(
        labels_dict=birds_catch["Bird1_bkd"],
        target_syl="bb",
        branch_syl = ["d", "i"],
        target_branch="d",
        figsize=(1.9, 1.5),
        save_dir=get_figure_dir("Figure1"),
        save_name="Fig1_D.svg",
        dpi=300
    )