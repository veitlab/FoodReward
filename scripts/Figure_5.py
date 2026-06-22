"""
Code for results depicted in Figure 5
    B) Syllable raster plot over multiple bouts
    C) Number of syllables following the target syllable in baseline vs. training
    D) Number of syllables following the target in catch vs. non-catch trials
    E) Number of syllables per bout in catch vs. non-catch trials

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
from matplotlib import colors as mcolors

from util.helper_fct import (
    load_birds,
    get_figure_dir,
    get_datasets,
    load_labels_with_feedback,
    BIRD_COLORS as bird_colors,
    BIRD_NAME_MAPPING as bird_name_mapping,
    TARGET_SYL_PER_BIRD as target_syl_per_bird,
    CONTEXT_PER_BIRD as context_per_bird,
    TARGET_REPEAT_PER_BIRD as target_repeat_per_bird,
    REPEATS_PER_BIRD as repeats_per_bird
)

from util.helper_fct import (
    transform_seq_for_plot, phase_from_day,
    syllables_after_target, group_align_positions_plot,
    build_syllable_color_map, build_matrix,
    extract_seq_counts, nanmean_or_nan,
    map_bird_name, setup_bs_t_summary_axis,
    sem_of_valid, stars_from_p,
    finalize_bs_t_summary_layout, day_sort_key,
    setup_catch_noncatch_summary_axis, 
    extract_song_lengths
)

#----- Load data -----#

birds = load_birds()
birds_catch = load_birds(catch=True)

datasets = get_datasets()

save_dir = get_figure_dir("Figure5")

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

def plot_bout_syllable_raster(
    labels_dict,
    sort_by="path",
    align_to_target=None,
    align_repeat_unit=None,
    align_repeat_index=None,
    max_bouts=None,
    inset_bouts=None,
    target_only=None,
    syllable_order=None,
    syllable_colors=None,
    figsize=(6, 8),
    inset_tick_step=10,
    show_align_marker=True,
    x_label=None,
    y_label="Song Index",
    baseline_title="B Target Bouts",
    training_title="T Target Bouts",
    save_dir=None,
    save_name=None,
    dpi=300,
):
    """
    Plots every bout in a single row, with syllables being depicted as color-coded rectangles.
    If no alignment is given, bouts are aligned to their start.

    Parameters
    ----------
    labels_dict : dict
        Dictionary of one experiment
    sort_by : str
        Sorts and plots bouts of baseline/ training either chronologically ("path"), by "length" or 
        by position of "target" in the song,
        default : "path"
    align_to_target : str or None
        Regex pattern defining the target motif used for bout selection and alignment,
        e.g. "(?:dc){4,}" selects bouts with at least four consecutive "dc" repeats.
        If None, bouts are plotted without target alignment.
    align_repeat_unit : str or None
        Repeat unit used to define a specific alignment position within the target motif,
        e.g. "dc" can be used to align to a specific "dc" repeat inside "(?:dc){4,}".
    align_repeat_index : int or None
        Index of the repeat unit to align to within the matched target motif.
        e.g. with align_repeat_unit="dc" and align_repeat_index=4,
        bouts are aligned to the fourth "dc" repeat.
    max_bouts : int or None
        Maximum number of bouts plotted per phase.
        If None, all matching bouts are plotted.
    inset_bouts : int or None
        Number of bouts shown in an inset-style plot.
        Uses the first baseline bouts and the last training bouts.
        If provided, this overrides max_bouts.
    target_only : bool or None
        If True, only bouts containing align_to_target are plotted.
        If None, this is automatically set to True when align_to_target is provided.
    syllable_order : list or None
        Custom order of syllables in the color map and legend
    syllable_colors : dict or None
        Optional mapping from syllables to fixed colors.
    inset_tick_step : int
        Tick spacing for inset-style plots.
    show_align_marker : bool
        If True, shows an arrow marking the alignment position
    x_label : str or None
        Label for the x-axis
    y_label : str
        Label for the y-axis
    baseline_title : str
        Title for the baseline panel
    training_title : str
        Title for the training panel
    """

    if target_only is None:
        target_only = align_to_target is not None

    records = []
    for file_path, seq_orig in labels_dict.items():
        if not (isinstance(seq_orig, str) and seq_orig):
            continue

        if align_to_target is not None and target_only:
            if re.search(align_to_target, seq_orig) is None:
                continue

        seq_plot, orig_to_plot = transform_seq_for_plot(seq_orig)

        records.append({
            "file_path": file_path,
            "phase": phase_from_day(file_path),
            "seq_orig": seq_orig,
            "seq_plot": seq_plot,
            "orig_to_plot": orig_to_plot,
        })

    if not records:
        raise ValueError("labels_dict enthaelt keine nicht-leeren passenden String-Sequenzen.")

    # sort bouts according to filter
    if sort_by == "length":
        records = sorted(records, key=lambda rec: (-len(rec["seq_orig"]), rec["file_path"]))
    elif sort_by == "target" and align_to_target is not None:
        def _target_sort_key(rec):
            match = re.search(align_to_target, rec["seq_orig"])
            if match is None:
                return (1, np.inf, rec["file_path"])
            return (0, match.start(), rec["file_path"])
        records = sorted(records, key=_target_sort_key)
    else:
        records = sorted(records, key=lambda rec: rec["file_path"])

    phase_groups = {
        "Baseline": [],
        "Training": [],
        "Postbaseline": [],
    }
    for rec in records:
        phase_groups[rec["phase"]].append(rec)

    phase_counts_before_slice = {k: len(v) for k, v in phase_groups.items()}

    if inset_bouts is not None:
        n = int(inset_bouts)
        if n < 1:
            raise ValueError("inset_bouts muss >= 1 sein.")
        phase_groups["Baseline"] = phase_groups["Baseline"][:n]
        phase_groups["Training"] = phase_groups["Training"][-n:]
        phase_groups["Postbaseline"] = phase_groups["Postbaseline"][:n]
    elif max_bouts is not None:
        max_bouts = int(max_bouts)
        for key in phase_groups:
            phase_groups[key] = phase_groups[key][:max_bouts]

    # Mann Whitney U test only for plotted bouts
    baseline_vals = []
    training_vals = []
    mwu_stat = np.nan
    mwu_p = np.nan

    if align_to_target is not None:
        for rec in phase_groups["Baseline"]:
            val = syllables_after_target(rec["seq_orig"], align_to_target)
            if np.isfinite(val):
                baseline_vals.append(val)

        for rec in phase_groups["Training"]:
            val = syllables_after_target(rec["seq_orig"], align_to_target)
            if np.isfinite(val):
                training_vals.append(val)

        baseline_vals = np.asarray(baseline_vals, dtype=float)
        training_vals = np.asarray(training_vals, dtype=float)

        if len(baseline_vals) > 0 and len(training_vals) > 0:
            mwu_stat, mwu_p = stats.mannwhitneyu(
                baseline_vals,
                training_vals,
                alternative="two-sided"
            )

            print("\n--- Mann–Whitney U Test (syllables after target; plotted bouts only) ---")
            print(f"Baseline: mean={np.mean(baseline_vals):.3f} (n={len(baseline_vals)}), + sem={np.std(baseline_vals, ddof=1) / np.sqrt(len(baseline_vals)):.3f}")
            print(f"Training: mean={np.mean(training_vals):.3f} (n={len(training_vals)}), + sem={np.std(training_vals, ddof=1) / np.sqrt(len(training_vals)):.3f}")
            print(f"U={mwu_stat:.3f}, p={mwu_p:.5g}, signif={mwu_p}")
        else:
            print("\nNicht genug Daten für Mann–Whitney U Test auf den geplotteten Bouts.")

    phase_row_numbers = {}
    for key in phase_groups:
        selected_n = len(phase_groups[key])
        full_n = phase_counts_before_slice[key]
        if inset_bouts is not None and key == "Training":
            start = max(1, full_n - selected_n + 1)
            phase_row_numbers[key] = list(range(start, full_n + 1))
        else:
            phase_row_numbers[key] = list(range(1, selected_n + 1))

    plotted_groups = []
    if phase_groups["Baseline"]:
        plotted_groups.append(("Baseline", baseline_title, phase_groups["Baseline"]))
    if phase_groups["Training"]:
        plotted_groups.append(("Training", training_title, phase_groups["Training"]))
    if not plotted_groups:
        fallback_title = "Songs with Target" if align_to_target is not None else "Songs"
        plotted_groups.append(("Postbaseline", fallback_title, phase_groups["Postbaseline"]))

    all_align_positions = []
    all_plot_lengths = []
    for _, _, group_records in plotted_groups:
        group_align_positions = group_align_positions_plot(group_records, align_to_target,
                                                            align_repeat_unit, align_repeat_index)
        all_align_positions.extend([pos for pos in group_align_positions if pos is not None])
        all_plot_lengths.extend([len(rec["seq_plot"]) for rec in group_records])

    if align_to_target is not None:
        if not all_align_positions:
            raise ValueError("align_to_target wurde gesetzt, aber in keinem geplotteten Bout gefunden.")

        left_pad = int(np.ceil(max(all_align_positions)))

        max_right = max(
            len(rec["seq_plot"]) - pos - 1
            for _, _, group_records in plotted_groups
            for rec, pos in zip(group_records, group_align_positions_plot(group_records, align_to_target,
                                                            align_repeat_unit, align_repeat_index))
            if pos is not None
        )
        n_cols = left_pad + int(np.ceil(max_right)) + 1
    else:
        left_pad = 0
        n_cols = max(all_plot_lengths)

    # only plot syllables that exist in plotted bouts 
    unique_syllables = []
    for _, _, group_records in plotted_groups:
        for rec in group_records:
            for syllable in rec["seq_plot"]:
                if syllable not in unique_syllables:
                    unique_syllables.append(syllable)

    if syllable_order is None:
        syllable_order = sorted(unique_syllables)
    else:
        syllable_order = [s for s in syllable_order if s in unique_syllables]
        syllable_order += [s for s in unique_syllables if s not in syllable_order]

    if syllable_colors is None:
        syllable_colors = build_syllable_color_map(syllable_order)
    else:
        syllable_colors = build_syllable_color_map(syllable_order, fixed_colors=syllable_colors)

    syllable_to_idx = {syllable: idx for idx, syllable in enumerate(syllable_order)}

    cmap = mcolors.ListedColormap([syllable_colors[s] for s in syllable_order])
    cmap.set_bad(color="white")
    norm = mcolors.BoundaryNorm(np.arange(len(syllable_order) + 1) - 0.5, cmap.N)

    n_panels = len(plotted_groups)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"hspace": 0.28},
    )
    if n_panels == 1:
        axes = [axes]

    panel_results = {}
    im = None

    for ax, (phase_name, title, group_records) in zip(axes, plotted_groups):
        matrix = build_matrix(group_records, align_to_target, align_repeat_unit, align_repeat_index,
                              n_cols, left_pad, syllable_to_idx)  
        im = ax.imshow(matrix, cmap=cmap, norm=norm, interpolation="none", aspect="auto")

        row_numbers = phase_row_numbers.get(phase_name, list(range(1, len(group_records) + 1)))
        panel_results[phase_name] = {
            "matrix": matrix,
            "bout_paths": [rec["file_path"] for rec in group_records],
            "row_numbers": row_numbers,
        }

        ax.set_ylabel(y_label)
        ax.set_title(title)

        n_rows = len(group_records)
        if n_rows <= 5:
            y_pos = list(range(n_rows))
        else:
            y_pos = np.linspace(0, n_rows - 1, 5).round().astype(int)
        y_lab = [str(row_numbers[i]) for i in y_pos]

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_lab)
        ax.tick_params(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if align_to_target is not None:
        for idx, ax in enumerate(axes):
            if idx == 0 and show_align_marker:
                ax.annotate(
                    "",
                    xy=(left_pad, 1.005),
                    xytext=(left_pad, 1.085),
                    xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                )

        if x_label is None:
            x_label = "Syllable Index"

        rel_min = -left_pad
        rel_max = (n_cols - 1) - left_pad
        tick_step = 20

        left_values = list(range(0, rel_min - 1, -tick_step))
        right_values = list(range(0, rel_max + 1, tick_step))
        rel_ticks = sorted(set(left_values + right_values))

        if 0 not in rel_ticks:
            rel_ticks.append(0)
            rel_ticks = sorted(rel_ticks)

        xticks = [t + left_pad for t in rel_ticks if 0 <= t + left_pad < n_cols]
        xticklabels = [str(t) for t in rel_ticks if 0 <= t + left_pad < n_cols]

        axes[-1].set_xticks(xticks)
        axes[-1].set_xticklabels(xticklabels)
    else:
        if x_label is None:
            x_label = "Syllable Index"
        axes[-1].tick_params(axis="x")

    axes[-1].set_xlabel(x_label)

    cbar = fig.colorbar(im, ax=axes, pad=0.02, fraction=0.05)
    cbar.set_ticks(np.arange(len(syllable_order)))
    display_labels = ["DC" if s == "D" else s for s in syllable_order]
    cbar.set_ticklabels(display_labels)
    cbar.set_label("Syllable")

    fig.tight_layout(rect=(0, 1, 1))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            suffix = "aligned" if align_to_target is not None else "plain"
            save_name = f"bout_syllable_raster_{suffix}.png"
        save_path = os.path.join(save_dir, save_name)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()

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
        # print(
        #     bird,
        #     "target:", target_syl_per_bird[bird],
        #     "repeat:", target_repeat,
        #     "n_base:", len(b_vals),
        #     "n_train:", len(t_vals),
        #     "base_mean:", nanmean_or_nan(b_vals),
        #     "train_mean:", nanmean_or_nan(t_vals)
        # )

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
    # print("Birds used in Wilcoxon:", [b for b, ok in zip(bird_names, valid_mask) if ok])
    # print("Birds excluded from Wilcoxon:", [b for b, ok in zip(bird_names, valid_mask) if not ok])
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

    x_base, x_train = setup_bs_t_summary_axis(ax, 1, 1)

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

    plt.show()

def syllables_after_target_catch_noncatch(
    birds,
    birds_catch,
    target_syl_per_bird,
    target_repeat_per_bird=None,
    n_last_train_days=3,
    bird_name_mapping=None,
    bird_colors=None,
    figsize=(5, 6),
    save_dir=None,
    save_name=None,
    dpi=300
):
    """
    Mean number of syllables following the target in catch vs. non-catch trials

    Parameters
    ----------
    birds : dict
        Dictionary containing non-catch trials from multiple experiments
    birds_catch : dict
        Dictionary containing catch trials from multiple experiments
    target_syl_per_bird : dict [str]
        The targeted repeat phrase for each experiment 
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

    if bird_name_mapping is None:
        bird_name_mapping = {}
    if bird_colors is None:
        bird_colors = {}
    bird_order = list(birds.keys())

    bird_names = []
    display_names = []
    catch_vals = []
    noncatch_vals = []
    detailed = {}

    for bird in bird_order:
        if bird not in birds_catch:
            print(f"[skip] {bird}: not in birds_catch dict")
            continue
        if bird not in target_syl_per_bird:
            print(f"[skip] {bird}: no target_syl defined")
            continue

        # Bestimme target_repeat für diesen Vogel
        if isinstance(target_repeat_per_bird, dict):
            trg_repeat = target_repeat_per_bird.get(bird, None)
        else:
            trg_repeat = target_repeat_per_bird

        training_days = sorted(
            {
                os.path.basename(os.path.dirname(path))
                for path in birds[bird].keys()
                if phase_from_day(os.path.basename(os.path.dirname(path))) == "Training"
            },
            key=day_sort_key
        )

        if n_last_train_days is not None:
            n_last_train_days = int(n_last_train_days)

            if n_last_train_days <= 0:
                training_days_used = []
            else:
                training_days_used = training_days[-n_last_train_days:]
        else:
            training_days_used = training_days

        training_days_used = set(training_days_used)

        noncatch_training_dict = {
            path: seq
            for path, seq in birds[bird].items()
            if os.path.basename(os.path.dirname(path)) in training_days_used
        }

        catch_training_dict = {
            path: seq
            for path, seq in birds_catch[bird].items()
            if os.path.basename(os.path.dirname(path)) in training_days_used
        }

        # Verwende _extract_seq_counts
        _, noncatch_vals_bird = extract_seq_counts(
            noncatch_training_dict,
            target_syl_per_bird[bird],
            position="after",
            target_repeat=trg_repeat,
        )

        _, catch_vals_bird = extract_seq_counts(
            catch_training_dict,
            target_syl_per_bird[bird],
            position="after",
            target_repeat=trg_repeat,
        )

        center_catch = nanmean_or_nan(np.array(catch_vals_bird, dtype=float))
        center_noncatch = nanmean_or_nan(np.array(noncatch_vals_bird, dtype=float))

        bird_names.append(bird)
        display_names.append(map_bird_name(bird, bird_name_mapping))
        catch_vals.append(center_catch)
        noncatch_vals.append(center_noncatch)

        detailed[bird] = {
            "target_syl": target_syl_per_bird[bird],
            "target_repeat": trg_repeat,
            "catch_raw": catch_vals_bird,
            "noncatch_raw": noncatch_vals_bird,
            "center_catch": center_catch,
            "center_noncatch": center_noncatch,
        }

    if len(bird_names) == 0:
        print("[warn] No birds to plot.")
        return {"birds": [], "details": {}}

    catch_arr = np.array(catch_vals, dtype=float)
    noncatch_arr = np.array(noncatch_vals, dtype=float)

    # --- Wilcoxon signed-rank test: catch vs non-catch ---
    valid_mask = np.isfinite(catch_arr) & np.isfinite(noncatch_arr)
    catch_valid = catch_arr[valid_mask]
    noncatch_valid = noncatch_arr[valid_mask]

    wilcoxon_stat = np.nan
    wilcoxon_p = np.nan
    wilcoxon_n = len(catch_valid)

    if wilcoxon_n >= 1:
        diffs = noncatch_valid - catch_valid
        n_nonzero = np.sum(diffs != 0)

        if n_nonzero > 0:
            try:
                res = stats.wilcoxon(
                    catch_valid,
                    noncatch_valid,
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
        f"Wilcoxon signed-rank test (catch vs non-catch): "
        f"n={wilcoxon_n}, statistic={wilcoxon_stat}, p={wilcoxon_p}"
    )

    wes_palette = [
        "#3B9AB2", "#78B7C5", "#E1AF00", "#EBCC2A",
        "#F21A00", "#9E7A7A", "#5F9D8A", "#C7B19C",
    ]
    colors = [
        bird_colors.get(bird, wes_palette[i % len(wes_palette)])
        for i, bird in enumerate(bird_names)
    ]

    fig, ax = plt.subplots(figsize=figsize)
    x_catch, x_noncatch = setup_catch_noncatch_summary_axis(ax, 1, 1)

    for i, bird in enumerate(bird_names):
        yc = catch_arr[i]
        ync = noncatch_arr[i]
        if np.isnan(yc) and np.isnan(ync):
            continue
        ax.plot(
            [x_catch, x_noncatch],
            [yc, ync],
            color=colors[i],
            lw=1,
            alpha=0.9,
            zorder=1,
            label=display_names[i]
        )

    # --- Group mean ± SEM ---
    group_catch = nanmean_or_nan(catch_arr)
    group_noncatch = nanmean_or_nan(noncatch_arr)
    group_catch_sem = sem_of_valid(catch_arr)
    group_noncatch_sem = sem_of_valid(noncatch_arr)

    if not (np.isnan(group_catch) and np.isnan(group_noncatch)):
        ax.errorbar(
            [x_catch, x_noncatch],
            [group_catch, group_noncatch],
            yerr=[group_catch_sem, group_noncatch_sem],
            color="black",
            lw=1.2,
            marker="o",
            markersize=4,
            capsize=3,
            elinewidth=1,
            alpha=1,
            zorder=5
        )

    ax.set_ylabel("n Syl. after Target")
    ax.set_ylim(0, )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y = 1.02
    h = 0.03
    ax.plot(
        [x_catch, x_catch, x_noncatch, x_noncatch],
        [y, y + h, y + h, y],
        transform=ax.get_xaxis_transform(),
        color="black",
        lw=1,
        clip_on=False
    )

    p_text = stars_from_p(wilcoxon_p)
    ax.text(
        (x_catch + x_noncatch) / 2,
        y + h,
        p_text,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        clip_on=False
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

    print(f"Catch: {group_catch:.2f} ± {group_catch_sem:.2f}, Non-Catch: {group_noncatch:.2f} ± {group_noncatch_sem:.2f}")
    print(f"Wilcoxon test: n={wilcoxon_n}, statistic={wilcoxon_stat}, p={wilcoxon_p}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = save_name if save_name else "summary_syl_after_target_catch_compare.png"
        save_path = os.path.join(save_dir, fname)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()

def song_length_catch_noncatch(
    birds_labels_dict,
    birds_labels_dict_catch,
    n_last_train_days=3,
    bird_name_mapping=None,
    bird_colors=None,
    figsize=(5, 6),
    save_dir=None,
    save_name=None,
    dpi=300
):
    """
    Mean number of syllables per bout in catch vs. non-catch trials

    Parameters
    ----------
    birds : dict
        Dictionary containing non-catch trials from multiple experiments
    birds_catch : dict
        Dictionary containing catch trials from multiple experiments
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
    bird_order = list(birds_labels_dict.keys())

    bird_names = []
    display_names = []
    catch_vals = []
    noncatch_vals = []
    detailed = {}

    for bird in bird_order:
        if bird not in birds_labels_dict:
            print(f"[skip] {bird}: not in birds_labels_dict")
            continue
        if bird not in birds_labels_dict_catch:
            print(f"[skip] {bird}: not in birds_labels_dict_catch")
            continue

        training_days = sorted(
            {
                os.path.basename(os.path.dirname(path))
                for path in birds_labels_dict[bird].keys()
                if phase_from_day(os.path.basename(os.path.dirname(path))) == "Training"
            },
            key=day_sort_key
        )

        n_last = None if n_last_train_days is None else int(n_last_train_days)

        if n_last is None:
            training_days_used = training_days
        elif n_last <= 0:
            training_days_used = []
        else:
            training_days_used = training_days[-n_last:]

        training_days_used = set(training_days_used)

        train_noncatch_dict = {
            path: seq
            for path, seq in birds_labels_dict[bird].items()
            if os.path.basename(os.path.dirname(path)) in training_days_used
        }

        train_catch_dict = {
            path: seq
            for path, seq in birds_labels_dict_catch[bird].items()
            if os.path.basename(os.path.dirname(path)) in training_days_used
        }

        train_noncatch = extract_song_lengths(train_noncatch_dict)
        train_catch = extract_song_lengths(train_catch_dict)

        center_noncatch = nanmean_or_nan(train_noncatch)
        center_catch = nanmean_or_nan(train_catch)

        bird_names.append(bird)
        display_names.append(map_bird_name(bird, bird_name_mapping))
        catch_vals.append(center_catch)
        noncatch_vals.append(center_noncatch)

        detailed[bird] = {
            "train_catch_lengths": train_catch,
            "train_noncatch_lengths": train_noncatch,
            "center_catch": center_catch,
            "center_noncatch": center_noncatch,
        }

    if len(bird_names) == 0:
        print("[warn] No birds to plot.")
        return {"birds": [], "details": {}}

    catch_vals = np.array(catch_vals, dtype=float)
    noncatch_vals = np.array(noncatch_vals, dtype=float)

    # --- Wilcoxon signed-rank test: catch vs non-catch ---
    valid_mask = np.isfinite(catch_vals) & np.isfinite(noncatch_vals)
    catch_valid = catch_vals[valid_mask]
    noncatch_valid = noncatch_vals[valid_mask]

    wilcoxon_stat = np.nan
    wilcoxon_p = np.nan
    wilcoxon_n = len(catch_valid)

    if wilcoxon_n >= 1:
        diffs = noncatch_valid - catch_valid
        n_nonzero = np.sum(diffs != 0)

        if n_nonzero > 0:
            try:
                res = stats.wilcoxon(
                    catch_valid,
                    noncatch_valid,
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
        f"Wilcoxon signed-rank test (catch vs non-catch): "
        f"n={wilcoxon_n}, statistic={wilcoxon_stat}, p={wilcoxon_p}"
    )

    fig, ax = plt.subplots(figsize=figsize)

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

    x_catch, x_noncatch = setup_catch_noncatch_summary_axis(ax, 1,1)

    for i, bird in enumerate(bird_names):
        color = colors[i]
        y_catch = catch_vals[i]
        y_noncatch = noncatch_vals[i]

        if np.isnan(y_catch) and np.isnan(y_noncatch):
            continue

        ax.plot(
            [x_catch, x_noncatch],
            [y_catch, y_noncatch],
            color=color,
            lw=1,
            alpha=0.9,
            zorder=1,
            label=display_names[i]
        )

    group_catch = nanmean_or_nan(catch_vals)
    group_noncatch = nanmean_or_nan(noncatch_vals)
    group_catch_sem = sem_of_valid(catch_vals)
    group_noncatch_sem = sem_of_valid(noncatch_vals)

    ax.errorbar(
        [x_catch, x_noncatch],
        [group_catch, group_noncatch],
        yerr=[group_catch_sem, group_noncatch_sem],
        color="black",
        lw=1.2,
        marker="o",
        markersize=4,
        elinewidth=1,
        capsize=3,
        alpha=1,
        zorder=5
        )

    ax.set_ylabel(r"n Syllables")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y = 1.02
    h = 0.03
    ax.plot(
        [x_catch, x_catch, x_noncatch, x_noncatch],
        [y, y + h, y + h, y],
        transform=ax.get_xaxis_transform(),
        color="black",
        lw=1,
        clip_on=False
    )

    p_text = stars_from_p(wilcoxon_p)
    ax.text(
        (x_catch + x_noncatch) / 2,
        y + h,
        p_text,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        clip_on=False
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            frameon=False,
            handlelength=0.6,
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
            save_name = "summary_song_length_catch_compare_across_birds.png"
        save_path = os.path.join(save_dir, save_name)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()

    print("Catch mean: {:.3f} ± {:.3f} (n={})".format(group_catch, group_catch_sem, np.sum(np.isfinite(catch_vals))))
    print("Non-Catch mean: {:.3f} ± {:.3f} (n={})".format(group_noncatch, group_noncatch_sem, np.sum(np.isfinite(noncatch_vals))))

#----- Run Main -----#

if __name__ == "__main__":

    # Fig 5 B
    # create color map
    rasterplot_colors = {
        "i":"#ffa734ff",
        "k":"#fffb21ff",
        "e":"#aff3f4ff",
        "d":"#eb53ffff",
        "c":"#db81d7ff",
        "D":"#1b35cfff",
        "a": "#da7474ff",
        "g": "#903dc7ff",
        "b": "#26d9e4ff",
        "f": "#C7B773",
        "j": "#6b96ffff",
        "l": "#E24023",
        "m":"#DDA044"
    }

    syllable_order = ["e", "b", "j", "D", "g", "d", "l", "m", "a", "i", "f", "k", "c"]

    plot_bout_syllable_raster(
        load_labels_with_feedback(datasets["Bird2_DC4"]),
        align_to_target="(?:dc){4,}",
        align_repeat_unit="dc",
        align_repeat_index=4,
        syllable_order=syllable_order,
        syllable_colors=rasterplot_colors,
        inset_bouts=20,
        figsize=(2.3, 2.6),
        save_dir=get_figure_dir("Figure5"),
        save_name="Fig5_B.svg",
        dpi=300
    )

    # Fig 5 C
    syllables_beforeafter_target(
        birds=birds,
        target_syl_per_bird=target_syl_per_bird,
        target_repeat_per_bird=target_repeat_per_bird,
        position="after",
        bird_name_mapping=bird_name_mapping,
        bird_colors=bird_colors,
        figsize=(1.525, 1.494),
        save_dir=get_figure_dir("Figure5"),
        save_name="Fig5_C.svg",
        dpi=300,
    )

    # Fig 5 D
    syllables_after_target_catch_noncatch(
        birds = birds,
        birds_catch=birds_catch,
        target_syl_per_bird=target_syl_per_bird,
        target_repeat_per_bird=target_repeat_per_bird,
        bird_name_mapping=bird_name_mapping,
        bird_colors=bird_colors,
        figsize=(1.525, 1.494),
        save_dir=get_figure_dir("Figure5"),
        save_name="Fig5_D.svg",
        dpi=300
    )

    # Fig 5 E
    song_length_catch_noncatch(
        birds_labels_dict=birds,
        birds_labels_dict_catch=birds_catch,
        bird_name_mapping=bird_name_mapping,
        bird_colors=bird_colors,
        figsize=(1.525, 1.494),
        save_dir=get_figure_dir("Figure5"),
        save_name="Fig5_E.svg",
        dpi=300
    )

