"""
Code for results depicted in Figure 4
    B) Multiple repeat distributions from the same dataset
    C) Target/ Non-target repeat change during training in all experiments
    D) Bouts / Hour in baseline vs. training
    E) Number of syllables in baseline vs. training
    F) Number of syllables before the target in baseline vs. training
    G) Number of target phrases per bout in baseline vs. training
    H) Appearance of the target repeat number per target phrase in baseline vs. training

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
from pathlib import Path
from matplotlib.lines import Line2D
from collections import defaultdict


from util.helper_fct import (
    load_birds,
    get_figure_dir,
    get_datasets,
    BIRD_COLORS as bird_colors,
    BIRD_NAME_MAPPING as bird_name_mapping,
    TARGET_SYL_PER_BIRD as target_syl_per_bird,
    CONTEXT_PER_BIRD as context_per_bird,
    TARGET_REPEAT_PER_BIRD as target_repeat_per_bird,
    REPEATS_PER_BIRD as repeats_per_bird
)

from util.helper_fct import (
    get_repeat_lengths, discrete_split_violin,
    choose_two_group_test, holm_correct_pvalues,
    stars_from_p, clean_repeat_label,
    get_plot_repeats, resolve_per_repeat_flags,
    day_sort_key, nanmean_or_nan, map_bird_name,
    finalize_bs_t_summary_layout, sem_of_valid,
    read_song_rate_day_data_from_experiment_csvs,
    split_baseline_training_days, setup_bs_t_summary_axis,
    compute_song_lengths_per_day, flatten_lengths,
    extract_seq_counts, compute_bouts_and_targets_per_day,
    compute_day_raw_values, get_lengths_from_vals,
    pooled_rate, flatten_repeats
)

#----- Load data -----#

birds = load_birds()
birds_catch = load_birds(catch=True)

datasets = get_datasets()

save_dir = get_figure_dir("Figure4")

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

def plot_mult_repeats(
    labels_dict,
    repeats,
    target_syl,
    context=False,
    highlight_colors=("#6e2222", "#942D2D"),
    bar_thickness=10,
    figsize=(8, 5),
    save_dir=None,
    save_name=None,
    dpi=300,
    training_test="auto"
):
    """
    Plots multiple repeats in the same data set, comparing baseline (left) and 
    training (right) in a discrete violin plot. 

    Tests the distribution of each repeat baseline vs. training

    Parameters
    ----------
    labels_dict : dict
        Dictionary of one experiment
    repeats : list(str)
        (Multiple) Repeats in this dataset
    target_syl : str
        The targeted repeat syllable, only for visualisation purpose
    context : bool
        bool for every individual repeat, True depicts context-dependency 

    Returns
    -------
    For each repeat: repeat distribution and song number per day, 
    results of the statistical tests
    """

    # collect repeats per day
    per_day_repeats = defaultdict(lambda: defaultdict(list))
    per_day_songs = defaultdict(int)

    for file_path, labels in labels_dict.items():
        if not labels:
            continue
        day_folder = os.path.basename(os.path.dirname(file_path))
        per_day_songs[day_folder] += 1

        if isinstance(context, np.ndarray):
            for repeat, contex in zip(repeats, context):
                lengths = get_repeat_lengths(labels, repeat, contex)
                per_day_repeats[day_folder][repeat].extend(lengths)
        else:
            for repeat in repeats:
                lengths = get_repeat_lengths(labels, repeat, context)
                per_day_repeats[day_folder][repeat].extend(lengths)

    baseline_data = defaultdict(list)
    training_data = defaultdict(list)

    for day, repeats_dict in per_day_repeats.items():
        for repeat, lengths in repeats_dict.items():
            if day.startswith("base"):
                baseline_data[repeat].extend(lengths)
            elif day.startswith("train"):
                training_data[repeat].extend(lengths)

    fig3, ax3 = plt.subplots(figsize=figsize)

    used_plot_repeats = []
    repeat_test_stats = {}
    repeat_to_x = {}

    if target_syl in repeats:
        plot_repeats = [target_syl] + [r for r in repeats if r != target_syl]
    else:
        plot_repeats = list(repeats)

    other_repeats = [r for r in plot_repeats if r != target_syl]

    repeat_specific_colors = {
        "(?:hf)+": ( "#154619", "#1F7426"),
        "[jm]+": ("#4F9442", "#7BE966"),
        "d+": ("#429472", "#59C497"),
        "e+": ("#679442", "#87C058"), 
        "b+":("#81B359", "#bae497ff"),        
        "kb+": highlight_colors,
        "ib+": ("#2f3030ff", "#555858ff"),
        "fb+": ("#5f6161ff", "#a1acacff"),
    }

    color_map = {}
    if target_syl in plot_repeats:
        color_map[target_syl] = highlight_colors

    for r in other_repeats:
        color_map[r] = ("#3a3b3bff", "#adb1b1ff")

    for r in plot_repeats:
        if r in repeat_specific_colors:
            color_map[r] = repeat_specific_colors[r]

    for i, repeat in enumerate(plot_repeats, start=1):
        base = np.array(baseline_data[repeat], dtype=float)
        train = np.array(training_data[repeat], dtype=float)

        if len(base) == 0 or len(train) == 0:
            print(f"[warn] No data for repeat {repeat}")
            continue

        repeat_to_x[repeat] = i

        col_base, col_train = color_map.get(repeat, ("0.4", "0.7"))
        violin_alpha = 1.0
        discrete_split_violin(
            ax3, i, base, train, 0.3,
            col_base, col_train,
            alpha=violin_alpha,
            line_width=bar_thickness
        )

        mean_base = float(np.mean(base))
        sem_base = float(np.std(base, ddof=1) / np.sqrt(len(base))) if len(base) > 1 else 0.0
        mean_train = float(np.mean(train))
        sem_train = float(np.std(train, ddof=1) / np.sqrt(len(train))) if len(train) > 1 else 0.0

        ax3.hlines(mean_base, i - 0.25, i - 0.05, color="black", lw=1, zorder=3)
        ax3.hlines(mean_train, i + 0.05, i + 0.25, color="black", lw=1, zorder=3)

        test_used, stat_val, p = choose_two_group_test(base, train, mode=training_test)

        used_plot_repeats.append(repeat)
        repeat_test_stats[repeat] = {
            "test": test_used,
            "stat": float(stat_val),
            "p_raw": float(p),
            "n_baseline": int(len(base)),
            "n_training": int(len(train)),
            "mean_baseline": mean_base,
            "mean_training": mean_train,
            "sem_baseline": sem_base,
            "sem_training": sem_train,
            "delta_mean": mean_train - mean_base,
        }

    if not used_plot_repeats:
        plt.close(fig3)
        print("No baseline/ training data found.")
        return per_day_repeats, per_day_songs, None

    pvalues_by_repeat = {
        repeat: repeat_test_stats[repeat]["p_raw"]
        for repeat in used_plot_repeats
    }
    corrected_p = holm_correct_pvalues(pvalues_by_repeat)

    print("\nRepeat statistics (baseline vs training):")
    for repeat in used_plot_repeats:
        repeat_test_stats[repeat]["p_corrected"] = corrected_p.get(repeat, np.nan)
        st = repeat_test_stats[repeat]
        stat_label = "t" if st["test"] == "welch_t" else "U"
        print(
            f"  {repeat:>10s} | "
            f"test={st['test']:<11s} | "
            f"baseline={st['mean_baseline']:.3f}±{st['sem_baseline']:.3f} (n={st['n_baseline']}) | "
            f"training={st['mean_training']:.3f}±{st['sem_training']:.3f} (n={st['n_training']}) | "
            f"Δ={st['delta_mean']:.3f} | "
            f"{stat_label}={st['stat']:.4f} | "
            f"p_raw={st['p_raw']:.4g} | "
            f"p_corr={st['p_corrected']:.4g}"
        )

    for i, repeat in enumerate(used_plot_repeats, start=1):
        base = np.array(baseline_data[repeat], dtype=float)
        train = np.array(training_data[repeat], dtype=float)
        y_max = max(base.max(), train.max())
        y_star = y_max + 0.1 * abs(y_max) + 0.2

        p_corr = repeat_test_stats[repeat]["p_corrected"]
        label = stars_from_p(p_corr)
        ax3.text(i, y_star, label, ha="center", va="bottom")

    clean_labels = [clean_repeat_label(r) for r in used_plot_repeats]
    ax3.set_xticks(range(1, len(used_plot_repeats) + 1))
    ax3.set_xticklabels(clean_labels)

    y_lo, y_hi = ax3.get_ylim()
    ax3.set_yticks(range(0, int(np.floor(y_hi)) + 1, 2))
    ax3.tick_params(axis="y")
    ax3.set_ylabel("Repeat Number")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig3.tight_layout(rect=[0, 0, 1, 0.93])

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            safe_target = re.sub(r"[^A-Za-z0-9]+", "_", target_syl).strip("_")
            save_name = f"mult_repeats_{safe_target}.png"
        save_path = os.path.join(save_dir, save_name)
        fig3.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()

    return per_day_repeats, per_day_songs, {
        "repeat_test_stats": repeat_test_stats,
        }

def summary_allrepeat_changes(
    birds_labels_dict,
    target_syl_per_bird,
    repeats_per_bird=None,
    context_per_bird=None,
    n_last_training_days=3,
    bird_name_mapping=None,
    figsize=(6, 6),
    save_dir=None,
    save_name=None,
    dpi=300
):
    """
    Summary change over all repeats in all experiments (targeted and non-targeted),
    depicting training mean repeat number (of n last days) relative to baseline value 
    for each repeat phrase.

    Parameters
    ----------
    birds_labels_dict: dict
        Dictionary containing label sequences from multiple experiments over multiple days
    target_syl_per_bird : dict [str]
        The targeted repeat phrase for each experiment 
    repeats_per_bird : dict [list[str]]
        All other repeat phrases that can appear in the bouts of this experiment,
        not the targeted phrase
    context_per_bird : dict [bool]
        For each target syllable whether it is context-dependent, True depicts context-dependency
        (Here, context-dependency is only considered for the target syllables!)
    n_last_train_days : int
        Number of last training days to consider,
        default : 3
    bird_name_mapping: dict
        Maps experiment names to more concise letters for the legend
    """

    if repeats_per_bird is None:
        repeats_per_bird = {}
    if context_per_bird is None:
        context_per_bird = {}
    if bird_name_mapping is None:
        bird_name_mapping = {}

    mwu_stat = np.nan
    mwu_p = np.nan

    bird_order = list(birds_labels_dict.keys())

    points = []
    details = {}

    per_repeat_tests = []
    n_larger = 0
    n_smaller = 0
    n_nonsig = 0
    n_tested = 0

    print("\n--- Per (bird × repeat) Baseline vs Training tests (Mann–Whitney U) ---")

    for bird in bird_order:
        if bird not in birds_labels_dict:
            print(f"[skip] {bird}: not in birds_labels_dict")
            continue
        if bird not in target_syl_per_bird:
            print(f"[skip] {bird}: no target in target_syl_per_bird")
            continue

        target_syl = target_syl_per_bird[bird]

        reps = list(dict.fromkeys(get_plot_repeats(target_syl, repeats_per_bird.get(bird, None))))

        context_spec = context_per_bird.get(bird, False)
        context_flags = resolve_per_repeat_flags(reps, context_spec, default=False)

        labels_data = birds_labels_dict[bird]

        train_day_names = sorted(
            {
                os.path.basename(os.path.dirname(p))
                for p in labels_data.keys()
                if os.path.basename(os.path.dirname(p)).lower().startswith("train")
            },
            key=day_sort_key
        )

        if n_last_training_days is not None:
            train_day_names = train_day_names[-n_last_training_days:]
        train_day_set = set(train_day_names)

        bird_detail = {
            "target_syl": target_syl,
            "training_catch_days_used": train_day_names,
            "repeats": {}
        }

        baseline_cache = {}
        train_cache = {}

        base_file_to_day = {}
        for file_path in labels_data.keys():
            day_name = os.path.basename(os.path.dirname(file_path))
            if day_name.lower().startswith("base"):
                base_file_to_day[file_path] = day_name

        train_file_to_day = {}
        for file_path in labels_data.keys():
            day_name = os.path.basename(os.path.dirname(file_path))
            if day_name in train_day_set:
                train_file_to_day[file_path] = day_name

        for file_path, seq in labels_data.items():
            if file_path not in base_file_to_day:
                continue

            for rep in reps:
                rep_context = context_flags.get(rep, False)

                lengths = get_repeat_lengths(seq, rep, rep_context)

                key = (rep, rep_context, base_file_to_day[file_path])
                if key not in baseline_cache:
                    baseline_cache[key] = []
                baseline_cache[key].extend(lengths)

        for file_path, seq in labels_data.items():
            if file_path not in train_file_to_day:
                continue

            for rep in reps:
                rep_context = context_flags.get(rep, False)

                lengths = get_repeat_lengths(seq, rep, rep_context)

                key = (rep, rep_context, train_file_to_day[file_path])
                if key not in train_cache:
                    train_cache[key] = []
                train_cache[key].extend(lengths)

        for rep in reps:
            rep_context = context_flags.get(rep, False)

            baseline_lengths = []
            baseline_days_used = set()

            for (r, ctx, day), lengths in baseline_cache.items():
                if r == rep and ctx == rep_context:
                    baseline_lengths.extend(lengths)
                    baseline_days_used.add(day)

            train_catch_lengths = []

            for (r, ctx, day), lengths in train_cache.items():
                if r == rep and ctx == rep_context:
                    train_catch_lengths.extend(lengths)

            base_mean = nanmean_or_nan(baseline_lengths)
            train_catch_last_mean = nanmean_or_nan(train_catch_lengths)

            base_str = f"{base_mean:.3f}" if np.isfinite(base_mean) else "nan"
            train_str = f"{train_catch_last_mean:.3f}" if np.isfinite(train_catch_last_mean) else "nan"
            print(
                f"[repeat-summary] bird={bird} repeat={rep} "
                f"baseline_mean={base_str} (n={len(baseline_lengths)}) "
                f"training_mean={train_str} (n={len(train_catch_lengths)})"
            )

            base_vals = np.asarray(baseline_lengths, dtype=float)
            train_vals = np.asarray(train_catch_lengths, dtype=float)

            base_vals = base_vals[np.isfinite(base_vals)]
            train_vals = train_vals[np.isfinite(train_vals)]

            if len(base_vals) > 0 and len(train_vals) > 0:
                stat, p = stats.mannwhitneyu(base_vals, train_vals, alternative="two-sided")

                mean_base = float(np.mean(base_vals))
                mean_train = float(np.mean(train_vals))

                if p < 0.05:
                    if mean_train > mean_base:
                        direction = "larger"
                        n_larger += 1
                    elif mean_train < mean_base:
                        direction = "smaller"
                        n_smaller += 1
                    else:
                        direction = "n.s."
                        n_nonsig += 1
                else:
                    direction = "n.s."
                    n_nonsig += 1

                n_tested += 1

                per_repeat_tests.append({
                    "bird": bird,
                    "repeat": rep,
                    "is_target": (rep == target_syl),
                    "U": float(stat),
                    "p": float(p),
                    "direction": direction,
                    "n_baseline": int(len(base_vals)),
                    "n_training": int(len(train_vals)),
                    "baseline_mean": mean_base,
                    "training_mean": mean_train,
                })
            else:
                print(f"[MWU] bird={bird} repeat={rep} | not enough data")

            points.append({
                "bird": bird,
                "display_name": map_bird_name(bird, bird_name_mapping),
                "repeat": rep,
                "is_target": (rep == target_syl),
                "x": base_mean,
                "y": train_catch_last_mean,
                "marker": "X" if (rep == target_syl) else "o",
            })

            bird_detail["repeats"][rep] = {
                "baseline_days_used": sorted(baseline_days_used, key=day_sort_key),
                "n_repeats_baseline": len(baseline_lengths),
                "n_repeats_training_catch_lastn": len(train_catch_lengths),
                "baseline_mean": base_mean,
                "training_catch_lastn_mean": train_catch_last_mean,
                "context": rep_context,
            }

        details[bird] = bird_detail

    if len(points) == 0:
        print("[warn] No birds to plot.")
        return {"birds": [], "details": {}}

    x_baseline = np.array([p["x"] for p in points], dtype=float)
    y_train_catch_last = np.array([p["y"] for p in points], dtype=float)

    relative_train_percent = np.full(len(points), np.nan, dtype=float)

    for i, p in enumerate(points):
        xb = p["x"]
        yt = p["y"]
        rel = relative_train_percent[i]

    for i, p in enumerate(points):
        xb = p["x"]
        yt = p["y"]
        if np.isfinite(xb) and np.isfinite(yt) and xb != 0:
            relative_train_percent[i] = 100.0 * yt / xb

    bird_names = []
    seen_birds = set()
    for p in points:
        if p["bird"] not in seen_birds:
            seen_birds.add(p["bird"])
            bird_names.append(p["bird"])

    fig_rel, ax_rel = plt.subplots(figsize=figsize)

    x_base_rel, x_train_rel = 0, 1

    relative_train_values = []
    for p in points:
        xb = p["x"]
        yt = p["y"]
        if np.isfinite(xb) and np.isfinite(yt) and xb != 0:
            yr = 1.0 + (yt - xb) / xb
        else:
            yr = np.nan
        relative_train_values.append(yr)

    relative_train_values = np.array(relative_train_values, dtype=float)

    finite_rel = relative_train_values[np.isfinite(relative_train_values)]
    if len(finite_rel) == 0:
        rel_min, rel_max = 0.5, 1.5
    else:
        vals = np.r_[finite_rel, 1.0]
        pad = 0.05 * (np.max(vals) - np.min(vals) + 1e-9)
        rel_min = np.min(vals) - pad
        rel_max = np.max(vals) + pad

    ax_rel.axhline(1.0, linestyle="--", color="black", lw=0.8, alpha=1)

    for i, p in enumerate(points):
        yr = relative_train_values[i]
        if not np.isfinite(yr):
            continue

        ax_rel.plot(
            [x_base_rel, x_train_rel],
            [1.0, yr],
            linewidth=0.3,
            linestyle="-",
            color="#dc5fe0ff" if p["is_target"] else "grey",
            alpha=1,
            zorder=1
        )

        ax_rel.scatter(
            x_base_rel,
            1.0,
            s=5,
            color="black",
            marker="o",
            edgecolor="none",
            linewidth=0.15 if p["is_target"] else 0.0,
            zorder=5 if p["is_target"] else 3,
        )

    target_vals = np.array([
        yr for yr, p in zip(relative_train_values, points)
        if np.isfinite(yr) and p["is_target"]
    ], dtype=float)

    non_target_vals = np.array([
        yr for yr, p in zip(relative_train_values, points)
        if np.isfinite(yr) and not p["is_target"]
    ], dtype=float)

    if len(target_vals) > 0:
        g_rel_target = np.nanmean(target_vals)
        sem_rel_target = (
            np.nanstd(target_vals, ddof=1) / np.sqrt(len(target_vals))
            if len(target_vals) > 1 else 0.0
        )

        label = "Target Repeats" if "Target Repeats" not in [l.get_label() for l in ax_rel.get_lines()] else None

        ax_rel.plot(
            [x_base_rel, x_train_rel],
            [1.0, g_rel_target],
            color="#dc5fe0ff",
            linewidth=1.0,
            alpha=1.0,
            zorder=6,
            label=label
        )

        ax_rel.scatter(
            [x_base_rel, x_train_rel],
            [1.0, g_rel_target],
            s=4,
            color="#dc5fe0ff",
            marker="o",
            zorder=7
        )
        ax_rel.errorbar(
            x_train_rel,
            g_rel_target,
            yerr=sem_rel_target,
            fmt="none",
            ecolor="#dc5fe0ff",
            elinewidth=1.2,
            capsize=3,
            zorder=7
        )

    if len(non_target_vals) > 0:
        g_rel_non = np.nanmean(non_target_vals)
        sem_rel_non = (
            np.nanstd(non_target_vals, ddof=1) / np.sqrt(len(non_target_vals))
            if len(non_target_vals) > 1 else 0.0
        )

        label = "Non-Target Repeats" if "Non-Target Repeats" not in [l.get_label() for l in ax_rel.get_lines()] else None

        ax_rel.plot(
            [x_base_rel, x_train_rel],
            [1.0, g_rel_non],
            color="k",
            linewidth=1.0,
            alpha=1.0,
            zorder=4,
            label=label
        )
        ax_rel.scatter(
            [x_base_rel, x_train_rel],
            [1.0, g_rel_non],
            s=4,
            color="k",
            marker="o",
            zorder=5
        )
        ax_rel.errorbar(
            x_train_rel,
            g_rel_non,
            yerr=sem_rel_non,
            fmt="none",
            ecolor="k",
            elinewidth=1.2,
            capsize=3,
            zorder=5
        )

    target_vals = np.array([
        yr for yr, p in zip(relative_train_values, points)
        if np.isfinite(yr) and p["is_target"]
    ], dtype=float)

    non_target_vals = np.array([
        yr for yr, p in zip(relative_train_values, points)
        if np.isfinite(yr) and not p["is_target"]
    ], dtype=float)


    if len(target_vals) > 0 and len(non_target_vals) > 0:
        try:
            mwu_res = stats.mannwhitneyu(
                target_vals,
                non_target_vals,
                alternative="two-sided"
            )
            mwu_stat = float(mwu_res.statistic)
            mwu_p = float(mwu_res.pvalue)
        except ValueError as e:
            print(f"Mann-Whitney U could not be computed: {e}")

        print("\n--- Target vs Non-target relative change ---")
        print(
            f"Target: mean={np.mean(target_vals):.4f}, n={len(target_vals)} | "
            f"Non-target: mean={np.mean(non_target_vals):.4f}, n={len(non_target_vals)}, sem={np.nanstd(non_target_vals, ddof=1) / np.sqrt(len(non_target_vals)):.4f}"
        )
        print(
            f"Mann-Whitney U: U={mwu_stat:.4f}, p={mwu_p:.6g}, "
            f"signif={stars_from_p(mwu_p)}"
        )

    if np.isfinite(mwu_p):
        anno_text = f"Target vs non-target: {stars_from_p(mwu_p)}"
    else:
        anno_text = "Target vs non-target: n.s."

    ax_rel.text(
        0.02, 0.98,
        anno_text,
        transform=ax_rel.transAxes,
        ha="left",
        va="top",
        color="#dc5fe0ff",
        zorder=30
    )

    ax_rel.set_xlim(-0.2, 1.2)
    ax_rel.set_ylim(rel_min, rel_max)
    ax_rel.set_xticks([x_base_rel, x_train_rel])
    ax_rel.set_xticklabels(["B", "T"])
    ax_rel.set_ylabel("Rel. Repeat Number")
    ax_rel.tick_params(axis="both")

    ax_rel.spines["top"].set_visible(False)
    ax_rel.spines["right"].set_visible(False)

    ax_rel.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
        handlelength=0.6,
    )

    finalize_bs_t_summary_layout(fig_rel)

    pos = ax_rel.get_position()
    scale = 0.667
    ax_rel.set_position([
        pos.x0 + pos.width * (1 - scale) / 2,
        pos.y0 + pos.height * (1 - scale) / 2,
        pos.width * scale,
        pos.height * scale
    ])

    mean_rel = np.nanmean(relative_train_values)
    sem_rel = np.nanstd(relative_train_values, ddof=1) / np.sqrt(np.sum(np.isfinite(relative_train_values)))
    print(f"Mean relative change (T/B): {mean_rel:.3f} ± {sem_rel:.3f}")

    mean_percent = (mean_rel - 1.0) * 100
    sem_percent = sem_rel * 100
    print(f"Mean % change: {mean_percent:.2f}% ± {sem_percent:.2f}%")

    target_vals = np.array([
        yr for yr, p in zip(relative_train_values, points)
        if np.isfinite(yr) and p["is_target"]
    ])
    mean_rel_t = np.nanmean(target_vals)
    sem_rel_t = np.nanstd(target_vals, ddof=1) / np.sqrt(len(target_vals))
    print(f"[TARGET] Mean % change: {(mean_rel_t-1)*100:.2f}% ± {sem_rel_t*100:.2f}%")
    
    if len(non_target_vals) > 0:
        mean_rel_nt = np.nanmean(non_target_vals)
        sem_rel_nt = (
            np.nanstd(non_target_vals, ddof=1) / np.sqrt(len(non_target_vals))
            if len(non_target_vals) > 1 else 0.0
        )
    else:
        mean_rel_nt = np.nan
        sem_rel_nt = np.nan
    print(f"[NON-TARGET] Mean % change: {(mean_rel_nt-1)*100:.2f}% ± {sem_rel_nt*100:.2f}%")
        
    print("\n--- Relative change summary ---")
    print(f"All repeats: {(mean_rel-1)*100:.2f}% ± {sem_percent:.2f}% (n={len(relative_train_values)})")
    print(f"Target only: {(mean_rel_t-1)*100:.2f}% ± {sem_rel_t*100:.2f}% (n={len(target_vals)})")
    print(f"Non-target only: {(mean_rel_nt-1)*100:.2f}% ± {sem_rel_nt*100:.2f}% (n={len(non_target_vals)})")

    print("\n--- Per (bird × repeat) test summary ---")
    print(f"{n_larger}/{n_tested} become significantly larger")
    print(f"{n_smaller}/{n_tested} become significantly smaller")
    print(f"{n_nonsig}/{n_tested} no significant change")

    target_tests = [t for t in per_repeat_tests if t["is_target"]]
    nontarget_tests = [t for t in per_repeat_tests if not t["is_target"]]

    def summarize(label, tests):
        if len(tests) == 0:
            return
        larger = sum(1 for t in tests if t["direction"] == "larger")
        smaller = sum(1 for t in tests if t["direction"] == "smaller")
        nonsig = sum(1 for t in tests if t["direction"] == "n.s.")

        print(f"\n--- {label} ---")
        print(f"{larger}/{len(tests)} larger")
        print(f"{smaller}/{len(tests)} smaller")
        print(f"{nonsig}/{len(tests)} n.s.")

    summarize("Target repeats", target_tests)
    summarize("Non-target repeats", nontarget_tests)

    if save_dir is not None:
        root, ext = os.path.splitext(save_name if save_name is not None else "summary_repeat_length_baseline_vs_catchlast_across_birds.png")
        if ext == "":
            ext = ".png"
        rel_name = f"{root}{ext}"
        rel_path = os.path.join(save_dir, rel_name)
        fig_rel.savefig(rel_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {rel_path}")

    plt.show()

def summary_song_rate_across_birds(
    root_dir_per_bird,
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
    Song rate in bouts per hour for all experiments, baseline vs. training.
    Song rate is calculated over the time the recording was running for each day,
    calculated from a separate batch file, containing also noise files.
    Songs are calculated from a batch file containing only song files.

    Parameters
    ----------
    root_dir_per_bird : dict
        Dictionary containing directories to the .csv files with the batch file information,
        need to be named _batch or _keep_notselect.
        When no _keep_notselect is found for one bird, _batch is used to calculate recording 
        duration and gain song number.
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

    bird_names = []
    baseline_vals = []
    training_vals = []

    baseline_total_songs = []
    training_total_songs = []
    baseline_total_hours = []
    training_total_hours = []

    for bird, root_dir in root_dir_per_bird.items():

        day_data = read_song_rate_day_data_from_experiment_csvs(
            root_dir_per_bird=root_dir_per_bird,
            experiment=bird,
            song_kind="keep_notselect",
            duration_kind="batch"
        )

        baseline_days, training_days = split_baseline_training_days(day_data)
        baseline_days = sorted(baseline_days, key=lambda x: day_sort_key(x[0]))
        training_days = sorted(training_days, key=lambda x: day_sort_key(x[0]))

        if n_last_training_days is not None:
            training_days = training_days[-n_last_training_days:]

        base_rate, base_songs, base_hours = pooled_rate(baseline_days)
        train_rate, train_songs, train_hours = pooled_rate(training_days)

        bird_names.append(bird)
        baseline_vals.append(base_rate)
        training_vals.append(train_rate)

        baseline_total_songs.append(base_songs)
        training_total_songs.append(train_songs)
        baseline_total_hours.append(base_hours)
        training_total_hours.append(train_hours)

    baseline_vals = np.array(baseline_vals, dtype=float)
    training_vals = np.array(training_vals, dtype=float)

    baseline_total_songs = np.array(baseline_total_songs, dtype=float)
    training_total_songs = np.array(training_total_songs, dtype=float)
    baseline_total_hours = np.array(baseline_total_hours, dtype=float)
    training_total_hours = np.array(training_total_hours, dtype=float)

    display_bird_names = [map_bird_name(bird, bird_name_mapping) for bird in bird_names]

    print("\nSummary song rate across birds:")
    for i, bird in enumerate(bird_names):
        print(
            f"{display_bird_names[i]} ({bird}): "
            f"baseline={baseline_vals[i]:.4f} songs/h "
            f"({int(baseline_total_songs[i])} songs / {baseline_total_hours[i]:.2f} h), "
            f"training={training_vals[i]:.4f} songs/h "
            f"({int(training_total_songs[i])} songs / {training_total_hours[i]:.2f} h)"
        )


    # --- Wilcoxon signed-rank test auf gepaarten Vogelwerten ---
    valid_mask = np.isfinite(baseline_vals) & np.isfinite(training_vals)
    baseline_valid = baseline_vals[valid_mask]
    training_valid = training_vals[valid_mask]

    # Overall mean and SEM
    baseline_mean = nanmean_or_nan(baseline_vals)
    baseline_sem = sem_of_valid(baseline_vals)
    training_mean = nanmean_or_nan(training_vals)
    training_sem = sem_of_valid(training_vals)

    print(f"\nOverall baseline: {baseline_mean:.4f} ± {baseline_sem:.4f} songs/h (n={len(baseline_valid)})")
    print(f"Overall training: {training_mean:.4f} ± {training_sem:.4f} songs/h (n={len(training_valid)})")

    wilcoxon_stat = np.nan
    wilcoxon_p = np.nan
    n_pairs = len(baseline_valid)

    if n_pairs >= 1:
        diffs = training_valid - baseline_valid
        n_nonzero = np.sum(diffs != 0)

        if n_nonzero > 0:
            try:
                wilcoxon_res = stats.wilcoxon(
                    baseline_valid,
                    training_valid,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto"
                )
                wilcoxon_stat = float(wilcoxon_res.statistic)
                wilcoxon_p = float(wilcoxon_res.pvalue)
            except ValueError as e:
                print(f"Wilcoxon test could not be computed: {e}")
        else:
            print("Wilcoxon test not computed: all paired differences are exactly zero.")
    else:
        print("Wilcoxon test not computed: no valid baseline/training pairs.")

    print(
        f"Wilcoxon signed-rank test across birds: "
        f"n_pairs={n_pairs}, statistic={wilcoxon_stat}, p={wilcoxon_p}"
    )

    fig, ax = plt.subplots(figsize=figsize)

    x_base_pos, x_train_pos = setup_bs_t_summary_axis(ax, 1,1)
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

        if show_bird_labels and not np.isnan(training_vals[i]):
            ax.text(
                x_train[i] + 0.03,
                training_vals[i],
                display_bird_names[i],
                va="center"
            )

    group_base = nanmean_or_nan(baseline_vals)
    group_train = nanmean_or_nan(training_vals)

    sem_base = sem_of_valid(baseline_vals)
    sem_train = sem_of_valid(training_vals)
    
    ax.errorbar(
        [x_base_pos, x_train_pos],
        [group_base, group_train],
        yerr=[sem_base, sem_train],
        color="black",
        lw=1.2,
        marker="o",
        markersize=4,
        capsize=3,
        elinewidth=1,
        alpha=1,
        zorder=3
    )
    
    # Positionen in AXES-Koordinaten (0–1)
    y = 1.02   # knapp über dem Plot
    h = 0.03   # Höhe der Klammer

    # Klammer zeichnen
    ax.plot(
        [x_base_pos, x_base_pos, x_train_pos, x_train_pos],
        [y, y + h, y + h, y],
        transform=ax.get_xaxis_transform(),  # <-- KEY
        color="black",
        lw=1,
        clip_on=False
    )

    # p-Wert bestimmen
    p_text = stars_from_p(wilcoxon_p)

    # Text
    ax.text(
        (x_base_pos + x_train_pos) / 2,
        y + h,
        p_text,
        transform=ax.get_xaxis_transform(),  # <-- KEY
        ha="center",
        va="bottom",
        clip_on=False
    )

    ax.set_ylabel("Bouts / Hour")
    ax.set_ylim(0, )

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
        handlelength=0.6
    )

    finalize_bs_t_summary_layout(fig)
    pos = ax.get_position()
    scale = 0.669
    ax.set_position([
        pos.x0 + pos.width * (1 - scale) / 2,
        pos.y0 + pos.height * (1 - scale) / 2,
        pos.width * scale,
        pos.height * scale
    ])


    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            save_name = "summary_song_rate_across_birds.png"
        save_path = os.path.join(save_dir, save_name)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()

def summary_song_length_base_train(
    birds_labels_dict,
    bird_name_mapping=None,
    bird_colors=None,
    figsize=(5, 6),
    n_last_training_days=3,
    save_dir=None,
    save_name=None,
    dpi=300,
    ):
    """
    Mean number of syllables per bout in baseline vs. training for multiple experiments.

    Parameters
    ----------
    birds_labels_dict: dict
        Dictionary containing label sequences from multiple experiments over multiple days
    bird_name_mapping: dict
        Maps experiment names to more concise letters for the legend
    bird_colors : dict
        Maps a specific color to each experiment
    n_last_train_days : int
        Number of last training days to consider,
        default : 3
    """

    if bird_name_mapping is None:
        bird_name_mapping = {}
    if bird_colors is None:
        bird_colors = {}
    bird_order = list(birds_labels_dict.keys())

    bird_names = []
    display_names = []
    baseline_vals = []
    training_vals = []
    detailed = {}

    for bird in bird_order:

        day_data = compute_song_lengths_per_day(birds_labels_dict[bird])

        baseline_days, training_days = split_baseline_training_days(day_data)
        baseline_days = sorted(baseline_days, key=lambda x: day_sort_key(x[0]))
        training_days = sorted(training_days, key=lambda x: day_sort_key(x[0]))

        if n_last_training_days is not None:
            training_days = training_days[-n_last_training_days:]

        baseline_lengths = flatten_lengths(baseline_days)
        training_lengths = flatten_lengths(training_days)

        center_baseline = nanmean_or_nan(baseline_lengths)
        center_training = nanmean_or_nan(training_lengths)

        bird_names.append(bird)
        display_names.append(map_bird_name(bird, bird_name_mapping))
        baseline_vals.append(center_baseline)
        training_vals.append(center_training)

        detailed[bird] = {
            "baseline_lengths": baseline_lengths,
            "training_lengths": training_lengths,
            "center_baseline": center_baseline,
            "center_training": center_training,
            "baseline_days_used": [d for d, _ in baseline_days],
            "training_days_used": [d for d, _ in training_days],
        }

    if len(bird_names) == 0:
        print("[warn] No birds to plot.")
        return {"birds": [], "details": {}}

    baseline_vals = np.array(baseline_vals, dtype=float)
    training_vals = np.array(training_vals, dtype=float)

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

    x_base, x_train = setup_bs_t_summary_axis(ax, 1,1)

    for i, bird in enumerate(bird_names):
        color = colors[i]
        y_base = baseline_vals[i]
        y_train = training_vals[i]

        if np.isnan(y_base) and np.isnan(y_train):
            continue

        ax.plot(
            [x_base, x_train],
            [y_base, y_train],
            color=color,
            lw=1,
            alpha=0.9,
            zorder=1,
            label=display_names[i]
        )

    group_base = nanmean_or_nan(baseline_vals)
    group_train = nanmean_or_nan(training_vals)
    group_base_sem = sem_of_valid(baseline_vals)
    group_train_sem = sem_of_valid(training_vals)

    if not (np.isnan(group_base) and np.isnan(group_train)):
        ax.errorbar(
            [x_base, x_train],
            [group_base, group_train],
            yerr=[group_base_sem, group_train_sem],
            color="black",
            lw=1.2,
            marker="o",
            markersize=4,
            elinewidth=1,
            capsize=3,
            alpha=1,
            zorder=5
        )
        print("Group Baseline mean: {:.3f} ± {:.3f} (n={})".format(group_base, group_base_sem, np.sum(np.isfinite(baseline_vals))))
        print("Group Training mean: {:.3f} ± {:.3f} (n={})".format(group_train, group_train_sem, np.sum(np.isfinite(training_vals))))

    ax.set_ylabel(r"n Syllables")
    ax.set_ylim(25, )
    ax.set_yticks([25, 50, 75])
    ax.set_yticklabels(["25", "50", "75"])
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
        clip_on=False
    )

    p_text = stars_from_p(wilcoxon_p)

    ax.text(
        (x_base + x_train) / 2,
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
            save_name = "summary_song_length_baseline_vs_training_across_birds.png"
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

    bouts_baseline = np.array(bouts_baseline, dtype=float)
    bouts_training = np.array(bouts_training, dtype=float)
    repeats_baseline = np.array(repeats_baseline, dtype=float)
    repeats_training = np.array(repeats_training, dtype=float)

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
    x_base_pos, x_train_pos = setup_bs_t_summary_axis(ax, 1,1)
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

    plt.show()

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

        bird_names.append(bird)
        baseline_vals.append(base_fraction)
        training_vals.append(train_fraction)
        baseline_total_repeats.append(base_total)
        training_total_repeats.append(train_total)
        baseline_target_repeats.append(base_target)
        training_target_repeats.append(train_target)

    baseline_vals = np.array(baseline_vals, dtype=float)
    training_vals = np.array(training_vals, dtype=float)
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

    x_base_pos, x_train_pos = setup_bs_t_summary_axis(ax,1,1)

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

        if show_bird_labels and not np.isnan(training_vals[i]):
            ax.text(
                x_train_pos + 0.03,
                training_vals[i],
                display_bird_names[i],
                va="center"
            )

    # --- Group mean + SEM ---
    group_base = nanmean_or_nan(baseline_vals)
    group_train = nanmean_or_nan(training_vals)
    group_base_sem = sem_of_valid(baseline_vals)
    group_train_sem = sem_of_valid(training_vals)

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

    plt.show()

    print(f"BS: {group_base:.2f} ± {group_base_sem:.2f}")
    print(f"TR: {group_train:.2f} ± {group_train_sem:.2f}")

#----- Run Main -----#

if __name__ == "__main__":

    # Fig 4 B
    plot_mult_repeats(
        birds_catch["Bird4_HF4"],
        repeats = ["(?:hf)+","b+","e+","[jm]+","d+"], 
        target_syl="(?:hf)",
        figsize=(4.5, 3),
        save_dir=get_figure_dir("Figure4"),
        save_name="Fig4_B.svg",
        dpi=300
    )

    # Fig 4 C
    summary_allrepeat_changes(
        birds_labels_dict={bird : data
            for bird, data in birds_catch.items()
            if bird != "Bird1_bkd"},
        target_syl_per_bird= {bird : data
            for bird, data in target_syl_per_bird.items()
            if bird != "Bird1_bkd"},   
        repeats_per_bird= {bird : data
            for bird, data in repeats_per_bird.items()
            if bird != "Bird1_bkd"},     
        context_per_bird= {bird : data
            for bird, data in context_per_bird.items()
            if bird != "Bird1_bkd"},  
        bird_name_mapping = bird_name_mapping,
        figsize=(4.5, 3),
        save_dir=get_figure_dir("Figure4"),
        save_name="Fig4_C.svg",
        dpi=300
    )


    # Fig 4 D
    # to recruite bout / day information
    root_dir_per_bird = {
        name: Path(path).parent
        for name, path in datasets.items()
        if not name.endswith("_catch")
    }

    summary_song_rate_across_birds(
        root_dir_per_bird=root_dir_per_bird,
        n_last_training_days=3,
        bird_name_mapping=bird_name_mapping,
        bird_colors=bird_colors,
        figsize=(1.525, 1.494),
        save_dir=get_figure_dir("Figure4"),
        save_name="Fig4_D.svg",
        dpi=300
    )

    # Fig 4 E
    summary_song_length_base_train(
        birds_labels_dict=birds_catch,
        bird_name_mapping=bird_name_mapping,
        bird_colors=bird_colors,
        figsize=(1.525, 1.494),
        save_dir=get_figure_dir("Figure4"),
        save_name="Fig4_E.svg",
        dpi=300
    )
    
    # Fig 4 F
    syllables_beforeafter_target(
        birds=birds_catch,
        target_syl_per_bird=target_syl_per_bird,
        target_repeat_per_bird=target_repeat_per_bird,
        position="before",
        bird_name_mapping=bird_name_mapping,
        bird_colors=bird_colors,
        figsize=(1.525, 1.494),
        save_dir=get_figure_dir("Figure4"),
        save_name="Fig4_F.svg",
        dpi=300,
    )

    # Fig 4 G
    summary_target_repeat_per_bout(
        birds=birds_catch,
        target_syl_per_bird=target_syl_per_bird,
        bird_name_mapping=bird_name_mapping,
        bird_colors=bird_colors,
        figsize=(1.525, 1.494),
        save_dir=get_figure_dir("Figure4"),
        save_name="Fig4_G.svg",
        dpi=300
    )

    # Fig 4 H
    summary_targetrepeat_per_repeats(
        birds= {bird : data
                for bird, data in birds_catch.items()
                if bird != "Bird1_bkd"},
        target_syl_per_bird=target_syl_per_bird,
        target_repeat_per_bird=target_repeat_per_bird,
        context_per_bird=context_per_bird,
        bird_name_mapping=bird_name_mapping,
        bird_colors=bird_colors,
        figsize=(1.525, 1.494),
        save_dir=get_figure_dir("Figure4"),
        save_name="Fig4_H.svg",
        dpi=300
    )