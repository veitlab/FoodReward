"""
Helper functions for the code repository accompanying:
Birdsong Modification with Food Reward

Author: Franziska Heubach
Year:2026
"""

import os
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import colorsys
import networkx as nx
from matplotlib import colors as mcolors
from scipy.stats import chi2_contingency, fisher_exact, wilcoxon, mannwhitneyu

#----- Loading data for all experiments -----#

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
PUBLISH_CODE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PUBLISH_CODE_DIR / "data"
SEQUENCE_DIR = DATA_DIR / "sequences"

FIGURE_DIR = PUBLISH_CODE_DIR / "figures"

BIRDS = [
    "Bird1_bkd",
    "Bird2_J6",
    "Bird2_B3",
    "Bird2_DC4",
    "Bird3_bF5",
    "Bird3_kB4",
    "Bird4_HF4",
    "Bird4_J5",
]

BIRD_COLORS = {
    "Bird1_bkd": "orange",
    "Bird2_J6": "#6b96ffff",
    "Bird2_B3": "#26d9e4ff",
    "Bird2_DC4": "#1b35cfff",
    "Bird3_bF5": "red",
    "Bird3_kB4": "#942d2dff",
    "Bird4_HF4": "#1F7426",
    "Bird4_J5": "#7BE966",
}

BIRD_NAME_MAPPING = {
    "bird1": "B1",
    "bird2": "B2",
    "bird3": "B3",
    "bird4": "B4",
}

TARGET_SYL_PER_BIRD = {
    "Bird1_bkd": "bbd",
    "Bird2_J6": "j+",
    "Bird2_B3": "b+",
    "Bird2_DC4": "(?:dc)+",
    "Bird3_bF5": "bf+",
    "Bird3_kB4": "kb+",
    "Bird4_HF4": "(?:hf)+",
    "Bird4_J5": "[jm]+",
}

CONTEXT_PER_BIRD = {
    "Bird1_bkd": False,
    "Bird2_J6": False,
    "Bird2_B3": False,
    "Bird2_DC4": False,
    "Bird3_bF5": True,
    "Bird3_kB4": True,
    "Bird4_HF4": False,
    "Bird4_J5": False,
}

TARGET_REPEAT_PER_BIRD = {
    "Bird1_bkd": None,
    "Bird2_J6": 6,
    "Bird2_B3": 3,
    "Bird2_DC4": 4,
    "Bird3_bF5": 5,
    "Bird3_kB4": 4,
    "Bird4_HF4": 4,
    "Bird4_J5": 5,
}

REPEATS_PER_BIRD = {
    "Bird2_J6": ["b+", "e+", "(?:dc)+"],
    "Bird2_B3": ["j+", "e+", "(?:dc)+"],
    "Bird2_DC4": ["b+", "e+", "j+"],
    "Bird3_bF5": ["if+", "mf+", "b+"],
    "Bird3_kB4": ["fb+", "ib+", "f+"],
    "Bird4_HF4": ["b+", "e+", "[jm]+", "d+"],
    "Bird4_J5": ["b+", "e+", "(?:hf)+", "d+"],
}

def load_labels(csv_file):
    df = pd.read_csv(csv_file)

    labels_dict = {}

    for _, row in df.iterrows():
        fake_path = os.path.join(
            str(row["day_folder"]),
            str(row["file"])
        )
        labels_dict[fake_path] = str(row["label_sequence"])

    return labels_dict

def get_datasets():
    """
    Return all sequence CSV paths.
    """
    return {
        "Bird1_bkd": SEQUENCE_DIR / "Bird1" / "Training_bkd" / "Bird1_bkd_sequence.csv",
        "Bird1_bkd_catch": SEQUENCE_DIR / "Bird1" / "Training_bkd" / "Bird1_bkd_sequence_catch.csv",

        "Bird2_J6": SEQUENCE_DIR / "Bird2" / "Training_J6" / "Bird2_J6_sequence.csv",
        "Bird2_J6_catch": SEQUENCE_DIR / "Bird2" / "Training_J6" / "Bird2_J6_sequence_catch.csv",

        "Bird2_B3": SEQUENCE_DIR / "Bird2" / "Training_B3" / "Bird2_B3_sequence.csv",
        "Bird2_B3_catch": SEQUENCE_DIR / "Bird2" / "Training_B3" / "Bird2_B3_sequence_catch.csv",

        "Bird2_DC4": SEQUENCE_DIR / "Bird2" / "Training_DC4" / "Bird2_DC4_sequence.csv",
        "Bird2_DC4_catch": SEQUENCE_DIR / "Bird2" / "Training_DC4" / "Bird2_DC4_sequence_catch.csv",

        "Bird3_bF5": SEQUENCE_DIR / "Bird3" / "Training_bF5" / "Bird3_bF5_sequence.csv",
        "Bird3_bF5_catch": SEQUENCE_DIR / "Bird3" / "Training_bF5" / "Bird3_bF5_sequence_catch.csv",

        "Bird3_kB4": SEQUENCE_DIR / "Bird3" / "Training_kB4" / "Bird3_kB4_sequence.csv",
        "Bird3_kB4_catch": SEQUENCE_DIR / "Bird3" / "Training_kB4" / "Bird3_kB4_sequence_catch.csv",

        "Bird4_HF4": SEQUENCE_DIR / "Bird4" / "Training_HF4" / "Bird4_HF4_sequence.csv",
        "Bird4_HF4_catch": SEQUENCE_DIR / "Bird4" / "Training_HF4" / "Bird4_HF4_sequence_catch.csv",

        "Bird4_J5": SEQUENCE_DIR / "Bird4" / "Training_J5" / "Bird4_J5_sequence.csv",
        "Bird4_J5_catch": SEQUENCE_DIR / "Bird4" / "Training_J5" / "Bird4_J5_sequence_catch.csv",
    }

def load_all_labels():
    datasets = get_datasets()

    return {
        name: load_labels(path)
        for name, path in datasets.items()
    }

def load_labels_with_feedback(csv_file):
    """
    Load label sequences and keep:
    - all baseline files
    - all postbaseline files
    - only training files with has_feedback == 1

    This does not filter catch vs. non-catch.
    """
    df = pd.read_csv(csv_file)

    if "has_feedback" not in df.columns:
        raise ValueError(
            f"{csv_file} has no 'has_feedback' column."
        )

    day_lower = df["day_folder"].astype(str).str.lower()

    is_baseline = day_lower.str.startswith("base")
    is_postbaseline = day_lower.str.startswith("post")
    has_feedback = df["has_feedback"].astype(int) == 1

    df = df[has_feedback | is_baseline | is_postbaseline]

    labels_dict = {}

    for _, row in df.iterrows():
        fake_path = os.path.join(
            str(row["day_folder"]),
            str(row["file"])
        )
        labels_dict[fake_path] = str(row["label_sequence"])

    return labels_dict

def make_birds_dict(labels, catch=False):
    """
    Create bird dictionary from loaded labels.
    """
    if catch:
        return {
            bird: labels[f"{bird}_catch"]
            for bird in BIRDS
        }

    return {
        bird: labels[bird]
        for bird in BIRDS
    }

def load_birds(catch=False):
    labels = load_all_labels()
    return make_birds_dict(labels, catch=catch)

def get_figure_dir(*subfolders, create=True):
    """
    Return a figure directory inside publish_code/figures.

    Examples
    --------
    get_figure_dir()
    -> publish_code/figures

    get_figure_dir("figure1")
    -> publish_code/figures/figure1
    """
    figure_dir = FIGURE_DIR.joinpath(*subfolders)

    if create:
        figure_dir.mkdir(parents=True, exist_ok=True)

    return figure_dir

#----- Analysis / Plot helper functions -----#

# Sorts phases by baseline/ training/ postbaseline
def day_sort_key(day_name):
    low = day_name.lower()
    if low.startswith("base"):
        phase = 0
    elif low.startswith("train"):
        phase = 1
    elif low.startswith("post"):
        phase = 2
    else:
        phase = 3

    m = re.search(r"(\d+)$", low)
    num = int(m.group(1)) if m else 0
    return (phase, num, low)

def phase_from_day(day_name):
    d = day_name.lower()
    if d.startswith("base"):
        return "Baseline"
    if d.startswith("train"):
        return "Training"
    if d.startswith("postbase"):
        return "Postbaseline"
    return None

# Plots segments ordered by B/ T/ P, mean +- SEM
def plot_segments(
    ax,
    x,
    y,
    yerr,
    color,
    label=None,
    linestyle='-',
    lw=1,
    fill_alpha=0.3,
    line_zorder=3,
    fill_zorder=2
    ):
    valid = np.isfinite(y)
    if not np.any(valid):
        return

    segs = []
    in_seg = False
    for k in range(len(x)):
        if valid[k]:
            if not in_seg:
                seg_start = k
                in_seg = True
        else:
            if in_seg:
                segs.append((seg_start, k))
                in_seg = False
    if in_seg:
        segs.append((seg_start, len(x)))

    for s_idx, (s, e) in enumerate(segs):
        lbl = label if s_idx == 0 else None
        ax.plot(
            x[s:e],
            y[s:e],
            linestyle=linestyle,
            lw=lw,
            color=color,
            label=lbl,
            zorder=line_zorder,
        )
        if np.any(np.isfinite(yerr[s:e])):
            ax.fill_between(
                x[s:e],
                y[s:e] - yerr[s:e],
                y[s:e] + yerr[s:e],
                color=color,
                alpha=fill_alpha,
                zorder=fill_zorder,
            )

# Derive significance stars from p-value
def stars_from_p(p):
    if not np.isfinite(p):
        return "n.s."
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."

# Computes branchpoint probabilities per day 
def compute_branch_probs(labels_dict, target_syl, branch_syl, show_postbase=True):
    """
    Event-based pooling of branching probability for each day.
    Computes all target → branch events across all files for each day, 
    then calculates the probability of branching to each branch syllable.
    """

    all_folders = sorted(
        list({os.path.basename(os.path.dirname(p)) for p in labels_dict.keys()}),
        key=day_sort_key
    )
    if not show_postbase:
        all_folders = [f for f in all_folders if not f.lower().startswith("post")]

    n_days = len(all_folders)
    n_branches = len(branch_syl)

    # matrices with one row per day, one column per branch
    # final column stores all transitions not listed in branch_syl
    branching_matrix = np.full((n_days, n_branches + 1), np.nan)
    branching_sem = np.full((n_days, n_branches + 1), np.nan)

    target_pattern = re.escape(target_syl) + r"(.)"

    for i, day_folder in enumerate(all_folders):
        day_files = [
            p for p in labels_dict.keys()
            if os.path.basename(os.path.dirname(p)) == day_folder
        ]
        if not day_files:
            continue
        
        # counts[j] stores number of target -> branch_syl[j] events
        # counts[-1] stores number of target -> other events
        counts = np.zeros(n_branches + 1, dtype=int)
        total_followers = 0

        for p in day_files:
            seq = labels_dict[p]
            if not seq:
                continue

            all_branches = re.findall(target_pattern, seq)
            if not all_branches:
                continue

            total_followers += len(all_branches)

            # observed followers that are in branch_syl
            known_count = 0
            for j, branch in enumerate(branch_syl):
                c = all_branches.count(branch)
                counts[j] += c
                known_count += c

            # other transitions not included in branch_syl
            counts[-1] += max(0, len(all_branches) - known_count)

        if total_followers == 0:
            continue

        probs = counts / total_followers
        branching_matrix[i, :] = probs

        branching_sem[i, :] = np.sqrt(probs * (1 - probs) / total_followers)

    return branching_matrix, branching_sem, all_folders

# Counts target - branch transition events for specified folders, used for tests
def count_branch_events_for_folders(labels_dict, folders, target_syl, branch_syl):
    """
    Count target→branch transition events within selected folders.
    Finds every syllable that immediately follows `target_syl` and counts how
    often each branch in `branch_syl` occurs.

    Transitions to syllables not listed in `branch_syl` are pooled into an
    additional "other" category stored in the last element of `counts`.
    """
    counts = np.zeros(len(branch_syl) + 1, dtype=int)
    target_pattern = re.escape(target_syl) + r"(.)"

    for p, seq in labels_dict.items():
        folder = os.path.basename(os.path.dirname(p))
        if folder not in folders:
            continue
        if not seq:
            continue
        all_branches = re.findall(target_pattern, seq)
        if not all_branches:
            continue
        known_count = 0
        for j, branch in enumerate(branch_syl):
            c = all_branches.count(branch)
            counts[j] += c
            known_count += c

        counts[-1] += max(0, len(all_branches) - known_count)

    return counts

# Run 2x2 test, Chi2-contingency or Fisher's exact test depending on expected frequencies
def run_2x2_branch_test(
    labels_dict,
    folders_a,
    folders_b,
    target_syl,
    branch_syl,
    target_branch=None,
    label_a="A",
    label_b="B",
    dataset_name="main"
):
    """
    Compare the frequency of one target branch between two groups of folders.

    For each group, all target→branch events are pooled across the selected
    folders. The function then builds a 2x2 contingency table for the 
    transition target_syl→target_branch, and all other followers of target_syl.

    Depending on the expected frequencies, the function uses either
    Fisher's exact test or a chi-square contingency test.
    """

    if target_branch is None:
        print(f"[{dataset_name}] target_branch is None -> no 2x2 test.")
        return None

    if len(folders_a) == 0:
        print(f"[{dataset_name}] No folders in group A ({label_a}) -> no test.")
        return None

    if len(folders_b) == 0:
        print(f"[{dataset_name}] No folders in group B ({label_b}) -> no test.")
        return None

    # Count all target→branch events for group A and group B.
    # The last element of each count vector contains "other" transitions.
    counts_a_all = count_branch_events_for_folders(
        labels_dict,
        folders_a,
        target_syl,
        branch_syl
    )

    counts_b_all = count_branch_events_for_folders(
        labels_dict,
        folders_b,
        target_syl,
        branch_syl
    )

    try:
        target_idx = branch_syl.index(target_branch)
    except ValueError:
        print(f"[{dataset_name}] target_branch '{target_branch}' is not in branch_syl.")
        return None

    # Extract the number of target transitions in each group.
    a_target = int(counts_a_all[target_idx])
    b_target = int(counts_b_all[target_idx])

    # Total number of target_syl→any-follower transitions in each group.
    a_total = int(counts_a_all.sum())
    b_total = int(counts_b_all.sum())

    # Number of non-target transitions in each group.
    a_not = a_total - a_target
    b_not = b_total - b_target

    if a_total == 0 or b_total == 0:
        print(f"[{dataset_name}] Not enough data for 2x2 test ({label_a} vs {label_b}).")
        return None

    # 2x2 contingency table:
    # rows = groups, columns = target vs not-target.
    contingency = np.array([
        [a_target, a_not],
        [b_target, b_not]
    ], dtype=int)

    # First compute expected frequencies to decide whether Fisher's exact test
    # is more appropriate than a chi-square test.
    _, _, _, expected = chi2_contingency(
        contingency,
        correction=False
    )

    use_fisher = np.any(expected < 5)

    if use_fisher:
        stat_value, p = fisher_exact(
            contingency,
            alternative="two-sided"
        )
        test_name = "Fisher's exact test"
        stat_name = "oddsratio"
        dof = None

    else:
        stat_value, p, dof, expected = chi2_contingency(
            contingency,
            correction=False
        )
        test_name = "Chi-square contingency test"
        stat_name = "chi2"

    print(f"\n[{dataset_name}] 2x2 test: target branch vs. not-target")
    print(f"  target_branch: {target_branch}")
    print(f"  {label_a}: {folders_a}")
    print(f"  {label_b}: {folders_b}")
    print("  contingency table [[A_target, A_not], [B_target, B_not]]:")
    print(contingency)

    print(f"  {label_a}: {a_target}/{a_total} = {100 * a_target / a_total:.2f}%")
    print(f"  {label_b}: {b_target}/{b_total} = {100 * b_target / b_total:.2f}%")
    print(f"  selected test: {test_name}")

    if dof is None:
        print(
            f"  {stat_name} = {stat_value:.4f}, "
            f"p = {p:.6g}, signif = {stars_from_p(p)}"
        )
    else:
        print(
            f"  {stat_name} = {stat_value:.4f}, "
            f"dof = {dof}, p = {p:.6g}, signif = {stars_from_p(p)}"
        )

    return {
        "test_name": test_name,
        "p": float(p),
        "stat_name": stat_name,
        "stat_value": float(stat_value),
        "dof": dof,
        "contingency": contingency,
        "a_target": a_target,
        "a_total": a_total,
        "b_target": b_target,
        "b_total": b_total,
        "folders_a": folders_a,
        "folders_b": folders_b,
    }
   
# Add significance bracket between two x positions, with text above
def add_sig_bracket(ax, x1, x2, y_ax=1.02, h=0.03, text="*", lw=1):
    trans = ax.get_xaxis_transform()

    ax.plot(
        [x1, x1, x2, x2],
        [y_ax, y_ax + h, y_ax + h, y_ax],
        transform=trans,
        color="black",
        lw=lw,
        clip_on=False,
        zorder=20,
    )

    ax.text(
        (x1 + x2) / 2,
        y_ax + h + 0.01,
        text,
        transform=trans,
        ha="center",
        va="bottom",
        zorder=21
    )    

# Calculate standard error of the mean
def sem_of_valid(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if arr.size == 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))

# Shapiro-Wilk test for normality
def normality_from_values(values, alpha=0.05):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        n = values.size
        if n < 3:
            return {
                "n": n,
                "p": np.nan,
                "normal": False,
                "note": "n<3, Shapiro nicht aussagekraeftig",
            }
        try:
            _, pval = stats.shapiro(values)
            is_normal = bool(np.isfinite(pval) and pval >= alpha)
            return {
                "n": n,
                "p": float(pval),
                "normal": is_normal,
                "note": "",
            }
        except Exception as exc:
            return {
                "n": n,
                "p": np.nan,
                "normal": False,
                "note": f"Shapiro-Fehler: {exc}",
            }
    
# Derive repeat unit length from target syllable pattern
def get_unit_len(local_target_syl):
    if local_target_syl.startswith("(?:") and local_target_syl.endswith(")+"):
        syl = local_target_syl[3:-2]
    elif local_target_syl.startswith("[") and local_target_syl.endswith("]+"):
        return 1
    else:
        syl = local_target_syl.rstrip("+")
    return len(syl)        

# Computes repeat number, including meta-repeats and context-dependency
def get_repeat_lengths(seq, local_target_syl, local_context=False):
    if not seq:
        return []

    matches = re.findall(local_target_syl, seq)

    if not matches:
        return []
    
    unit_len = get_unit_len(local_target_syl)

    if local_context:
        return [len(m) - 1 for m in matches]
    else:
        return [len(m) // unit_len for m in matches]

# Returns raw repeat number values for each day, pooled across all files, used for mean repeat length
def compute_day_raw_values(labels_dict, target_syl, context=False):
    all_folders_local = sorted(
        {
            os.path.basename(os.path.dirname(p))
            for p in labels_dict.keys()
        },
        key=day_sort_key
    )

    day_to_values = {}

    for day_folder in all_folders_local:
        day_files = [
            p for p in labels_dict.keys()
            if os.path.basename(os.path.dirname(p)) == day_folder
        ]

        vals = []
        for p in day_files:
            seq = labels_dict[p]
            lengths = get_repeat_lengths(seq, target_syl, context)
            vals.extend(lengths)

        day_to_values[day_folder] = np.asarray(vals, dtype=float)

    return day_to_values    

# Returns mean + SEM of repeat number for each day  
def summarize_day_values(day_to_values):
    folders = sorted(day_to_values.keys(), key=day_sort_key)

    day_mean = np.full(len(folders), np.nan)
    day_sem = np.full(len(folders), np.nan)

    for i, folder in enumerate(folders):
        vals = np.asarray(day_to_values[folder], dtype=float)
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            continue

        day_mean[i] = np.nanmean(vals)
        day_sem[i] = sem_of_valid(vals)

    return day_mean, day_sem, folders  

# Pool day values into single array
def pool_all_lengths(day_items):
    all_lengths = []

    for _, vals in day_items:
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]

        if arr.size > 0:
            all_lengths.append(arr)

    if len(all_lengths) == 0:
        return np.array([], dtype=float)

    return np.concatenate(all_lengths)

# Maps bird names to shorter labels
def map_bird_name(bird_name, bird_name_mapping):
    parts = bird_name.split("_", 1)
    prefix = parts[0].lower()
    suffix = "_" + parts[1] if len(parts) > 1 else ""
    for key, replacement in bird_name_mapping.items():
        if prefix == key.lower():
            return replacement + suffix
    return bird_name

def nanmean_or_nan(values):
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return np.nan
    if np.all(np.isnan(values)):
        return np.nan
    return np.nanmean(values)

# Uniform axis for small line plots
def setup_bs_t_summary_axis(ax, label_fs, tick_fs):
    """Einheitliche x-Achse für alle Baseline-vs-Training-Summary-Plots."""
    x_base = 0.35
    x_train = 0.65
    ax.set_xticks([x_base, x_train])
    ax.set_xticklabels(["B", "T"])
    ax.tick_params(axis='y')
    ax.set_xlim(0.3, 0.7)
    return x_base, x_train

def setup_catch_noncatch_summary_axis(ax, label_fs, tick_fs):
    """Einheitliche x-Achse für Catch-vs-Non-Catch-Summary-Plots."""
    x_catch = 0.35
    x_noncatch = 0.65
    ax.set_xticks([x_catch, x_noncatch])
    ax.set_xticklabels(["C", "NC"])
    ax.tick_params(axis='y')
    ax.set_xlim(0.3, 0.7)
    return x_catch, x_noncatch

def finalize_bs_t_summary_layout(fig, right=0.78):
    """Feste Ränder, damit BS/T-Summary-Plots dieselbe sichtbare Plotbreite haben."""
    fig.subplots_adjust(left=0.24, right=right, bottom=0.18, top=0.96)
    
# Calculates pooled self-transition probability for baseline/ training
def pooled_p_self(labels_dict, target_syl, repeat_number, context, day_predicate):
    total_self = 0
    total_other = 0
    n_files_used = 0
    n_files_with_events = 0

    for file_path, seq in labels_dict.items():
        if not isinstance(seq, str) or not seq:
            continue

        day_name = os.path.basename(os.path.dirname(file_path))
        if not day_predicate(day_name):
            continue

        n_files_used += 1

        s_cnt, o_cnt = self_other_counts_in_seq(
            seq,
            target_syl=target_syl,
            repeat_number=repeat_number,
            context=context,
        )
        total_self += s_cnt
        total_other += o_cnt
        if (s_cnt + o_cnt) > 0:
            n_files_with_events += 1

    total = total_self + total_other
    if total == 0:
        return np.nan, 0, int(n_files_used), int(n_files_with_events)
    return float(total_self / total), int(total), int(n_files_used), int(n_files_with_events)

# Counts self-transition probability of target syl
def self_other_counts_in_seq(seq, target_syl, repeat_number, context=False):
    repeat_lengths = get_repeat_lengths(
        seq,
        local_target_syl=target_syl,
        local_context=context
    )

    self_count = 0
    other_count = 0

    for repeat_len in repeat_lengths:
        if repeat_len < (repeat_number - 1):
            continue
        if repeat_len >= repeat_number:
            self_count += 1
        else:
            other_count += 1

    return self_count, other_count

# Wilcoxon for every position in self-transition plot 
def wilcoxon_by_training_from_summary(summary_result, min_support_files=3):
    per_training = summary_result["per_bird"]
    offsets = summary_result["offsets"]

    rows = []
    details = {}

    for off in offsets:
        baseline_vals = []
        training_vals = []
        used_trainings = []

        for training_name, training_data in per_training.items():
            d = training_data[off]

            base = d.get("baseline_mean", np.nan)
            train = d.get("training_mean", np.nan)
            n_support = int(d.get("n_files_with_events_training", 0))

            if not np.isfinite(base) or not np.isfinite(train):
                continue
            if n_support < int(min_support_files):
                continue

            baseline_vals.append(float(base))
            training_vals.append(float(train))
            used_trainings.append(training_name)

        baseline_vals = np.asarray(baseline_vals, dtype=float)
        training_vals = np.asarray(training_vals, dtype=float)
        delta_vals = training_vals - baseline_vals if len(baseline_vals) else np.array([])

        n_trainings = len(baseline_vals)

        if n_trainings < 2:
            stat = np.nan
            pval = np.nan
        elif np.allclose(baseline_vals, training_vals, equal_nan=True):
            stat = 0.0
            pval = 1.0
        else:
            stat, pval = wilcoxon(
                baseline_vals,
                training_vals,
                zero_method="wilcox"
            )

        rows.append({
            "offset": off,
            "n_trainings": n_trainings,
            "baseline_mean": np.nanmean(baseline_vals) if n_trainings else np.nan,
            "training_mean": np.nanmean(training_vals) if n_trainings else np.nan,
            "mean_delta": np.nanmean(delta_vals) if n_trainings else np.nan,
            "wilcoxon_stat": stat,
            "pvalue": pval,
        })

        details[off] = {
            "used_trainings": used_trainings,
            "baseline_vals": baseline_vals,
            "training_vals": training_vals,
            "delta_vals": delta_vals,
        }

    results_df = pd.DataFrame(rows)
    return results_df, details

# finds target syl in mutliple repeats and puts it in front
def get_plot_repeats(target_syl, repeats=None):
    """
    target immer vorne. Wenn repeats None -> [target].
    """
    if repeats is None:
        return [target_syl]
    reps = list(repeats)
    if target_syl in reps:
        return [target_syl] + [r for r in reps if r != target_syl]
    return [target_syl] + reps

# creates dictionary of sorted repeat / context
def resolve_per_repeat_flags(repeats, flag_spec, default=False):
    """
    Normalisiert chunk/context Spezifikation zu dict: repeat -> bool

    flag_spec kann sein:
      - bool
      - list/np.ndarray aligned zu repeats
      - dict {repeat: bool}
      - None

    repeats: list in genau der Reihenfolge, in der du sie plottest.
    """
    if flag_spec is None:
        return {r: bool(default) for r in repeats}

    # bool für alle repeats
    if isinstance(flag_spec, (bool, np.bool_)):
        return {r: bool(flag_spec) for r in repeats}

    # dict pro repeat
    if isinstance(flag_spec, dict):
        return {r: bool(flag_spec.get(r, default)) for r in repeats}

    # list/array aligned
    if isinstance(flag_spec, (list, np.ndarray)):
        out = {}
        for r, v in zip(repeats, flag_spec):
            out[r] = bool(v)
        # falls Liste zu kurz: Rest default
        for r in repeats[len(out):]:
            out[r] = bool(default)
        return out

    # fallback
    return {r: bool(default) for r in repeats}

def read_song_rate_day_data_from_experiment_csvs(
    root_dir_per_bird,
    experiment,
    song_kind="keep_notselect",
    duration_kind="batch"
):
    root_dir = Path(root_dir_per_bird[experiment])

    def find_csv(kind):
        candidates = [
            root_dir / f"{experiment}_{kind}.csv",
            root_dir / f"{experiment}_{kind}",
            root_dir / f"{experiment}_{kind}_summary.csv",
        ]

        for path in candidates:
            if path.exists():
                return path

        return None

    duration_csv = find_csv(duration_kind)
    song_csv = find_csv(song_kind)

    if duration_csv is None:
        raise FileNotFoundError(
            f"No duration CSV found for {experiment} in {root_dir}"
        )

    # Bird1 fallback: if no keep_notselect CSV exists, use batch CSV for songs too
    if song_csv is None:
        song_csv = duration_csv

    duration_df = pd.read_csv(duration_csv)
    song_df = pd.read_csv(song_csv)

    duration_df["day"] = duration_df["day"].astype(str)
    song_df["day"] = song_df["day"].astype(str)

    duration_df = duration_df[
        duration_df["phase"].astype(str).str.lower().isin(["baseline", "training"])
    ].copy()

    song_df = song_df[
        song_df["phase"].astype(str).str.lower().isin(["baseline", "training"])
    ].copy()

    day_data = {}

    for _, dur_row in duration_df.iterrows():
        day = dur_row["day"]

        matching_song = song_df[song_df["day"] == day]

        if len(matching_song) == 0:
            n_songs = 0
        else:
            n_songs = int(matching_song.iloc[0]["n_timestamped_files"])

        duration_hours = float(dur_row["duration_hours"])

        day_data[day] = {
            "count": n_songs,
            "duration_hours": duration_hours,
        }

    return day_data

def split_baseline_training_days(day_dict):
    baseline_days = []
    training_days = []

    for day_name, vals in day_dict.items():
        low = day_name.lower()

        if low.startswith("base"):
            baseline_days.append((day_name, vals))
        elif low.startswith("train"):
            training_days.append((day_name, vals))

    return baseline_days, training_days

def compute_song_lengths_per_day(labels_dict):
    day_to_lengths = {}

    for p, labels in labels_dict.items():
        if not labels:
            continue

        day_name = os.path.basename(os.path.dirname(p))

        if day_name not in day_to_lengths:
            day_to_lengths[day_name] = []

        day_to_lengths[day_name].append(len(labels))

    return {
        day_name: {
            "lengths": np.array(lengths, dtype=float),
        }
        for day_name, lengths in day_to_lengths.items()
    }

def count_repeat_units_in_match(match_text, target_syl):
    """
    Count how often the repeated unit occurs inside one regex match.

    Examples
    --------
    target_syl="j+", match_text="jjj"         -> 3
    target_syl="bf+", match_text="bfff"       -> 3
    target_syl="kb+", match_text="kbbb"       -> 3
    target_syl="(?:dc)+", match_text="dcdc"   -> 2
    target_syl="[jm]+", match_text="jmj"      -> 3
    """

    if not isinstance(target_syl, str) or not target_syl.endswith("+"):
        return 1

    # Meta-repeat, e.g. (?:dc)+
    if target_syl.startswith("(?:") and target_syl.endswith(")+"):
        unit_len = get_unit_len(target_syl)
        return len(match_text) // unit_len

    # Character class, e.g. [jm]+
    if target_syl.startswith("[") and target_syl.endswith("]+"):
        return len(match_text)

    # Plain repeat, e.g. j+, bf+, kb+
    # j+  -> no prefix, repeated unit is j
    # bf+ -> prefix is b, repeated unit is f
    # kb+ -> prefix is k, repeated unit is b
    base = target_syl.rstrip("+")
    prefix_len = max(len(base) - 1, 0)

    return len(match_text) - prefix_len

def extract_seq_counts(
    labels_dict,
    target_syl,
    position,
    target_repeat=None,
    n_last_training_days=3
):
    """
    For each song, count the number of syllables before or after the first valid target.

    A valid target is:
    - the first match of target_syl, if target_repeat is None
    - the first match with repeat count >= target_repeat, if target_repeat is given

    Parameters
    ----------
    labels_dict : dict
        file_path -> label string

    target_syl : str
        Regex pattern for the target, e.g. "j+", "bf+", "(?:dc)+".

    position : str
        "before" -> count syllables before the target starts
        "after"  -> count syllables after the target ends

    target_repeat : int or None, default=None
        If given, only matches with repeat length >= target_repeat are used.

    Returns
    -------
    baseline_vals, training_vals : list, list
        Counts for baseline and training songs.
    """

    if position not in ("before", "after"):
        raise ValueError("position must be 'before' or 'after'")

    target_re = re.compile(target_syl)

    if target_repeat is not None:
        try:
            target_repeat = int(target_repeat)
        except (TypeError, ValueError):
            target_repeat = None

    # Determine which training days to use
    # Determine which training days to use
    all_training_days = sorted(
        {
            os.path.basename(os.path.dirname(file_path))
            for file_path in labels_dict.keys()
            if phase_from_day(os.path.basename(os.path.dirname(file_path))) == "Training"
        },
        key=day_sort_key
    )

    if n_last_training_days is not None:
        if n_last_training_days <= 0:
            training_day_set = set()
        else:
            training_day_set = set(all_training_days[-n_last_training_days:])
    else:
        training_day_set = set(all_training_days)

    if n_last_training_days is not None:
        if n_last_training_days <= 0:
            training_day_set = set()
        else:
            training_day_set = set(all_training_days[-n_last_training_days:])
    else:
        training_day_set = set(all_training_days)

    baseline_vals = []
    training_vals = []

    for file_path, labels in labels_dict.items():
        if not labels or not isinstance(labels, str):
            continue

        day_name = os.path.basename(os.path.dirname(file_path))
        phase = phase_from_day(day_name)

        if phase is None:
            continue
        if phase == "Training" and day_name not in training_day_set:
            continue

        selected_match = None

        for match in target_re.finditer(labels):
            if target_repeat is None or target_repeat <= 1:
                selected_match = match
                break

            repeat_count = count_repeat_units_in_match(
                match_text=match.group(0),
                target_syl=target_syl
            )

            if repeat_count >= target_repeat:
                selected_match = match
                break

        if selected_match is None:
            continue

        start, end = selected_match.span()

        if position == "before":
            value = start
        else:
            value = len(labels) - end

        if phase == "Baseline":
            baseline_vals.append(value)
        elif phase == "Training":
            training_vals.append(value)

    return baseline_vals, training_vals

def compute_bouts_and_targets_per_day(labels_dict, target_syl):
    """
    Berechnet pro Tag:
      - n_bouts: Anzahl Songs/WAV-Files mit Labels (ein Song = ein Bout)
      - repeats_per_song: Anzahl Target-Repeats pro Song (unabhängig von Repeat-Länge)
      - mean_repeats_per_song: Mittelwert der Repeat-Anzahl pro Song

    Jeder Regex-Match von target_syl zählt als ein Repeat-Event,
    egal wie lang der Match ist (z.B. jjj und jjjjj zählen jeweils als 1 Event).
    """
    day_to_data = {}
    all_days = sorted({os.path.basename(os.path.dirname(p)) for p in labels_dict.keys()})

    for day_folder in all_days:
        day_files = [
            p for p in labels_dict.keys()
            if os.path.basename(os.path.dirname(p)) == day_folder
        ]

        repeats_per_song = []
        n_bouts = 0

        for p in day_files:
            labels = labels_dict[p]
            if not labels:
                continue

            # Ein geladener Song/WAV entspricht hier einem Bout.
            n_bouts += 1

            # Jeder Match zählt als ein Repeat-Event, unabhängig von der Länge des Matches.
            n_target_repeats = len(re.findall(target_syl, labels))
            repeats_per_song.append(n_target_repeats)

        repeats_per_song = np.array(repeats_per_song, dtype=float)
        mean_repeats = nanmean_or_nan(repeats_per_song)

        day_to_data[day_folder] = {
            "n_bouts": n_bouts,
            "repeats_per_song": repeats_per_song,
            "mean_repeats_per_song": mean_repeats,
        }

    return day_to_data

def get_lengths_from_vals(vals):
    if isinstance(vals, dict):
        lengths = vals.get("lengths", [])
    else:
        lengths = vals

    lengths = np.asarray(lengths, dtype=float)
    lengths = lengths[np.isfinite(lengths)]

    return lengths

# find songs with feedback (real non-catch files, no misses)
def summarize_feedback_information(labels_dict, print_per_file=True, feedback_pattern=None):
    """
    Sucht fuer jeden Eintrag in labels_dict die passende .rec-Datei,
    extrahiert Feedback-Eintraege per Regex und trennt in
    'mit Match' vs. 'ohne Match'.

    Inhaltlich ist das eine reine Parsing-/Inspektionsfunktion für .rec-Dateien:
    Sie ändert labels_dict nicht, sondern liefert nur Zusatzinformation dazu,
    in welchen zugehörigen rec-Dateien Feedback-Ereignisse gefunden wurden.

    Returns
    -------
    dict mit Schluesseln:
        with_feedback, without_feedback, missing_rec, missing_pattern
    """
    def _rec_path_from_label_path(file_path):
        # Die Funktion arbeitet nur mit labels_dict-Keys. Daraus wird heuristisch
        # der erwartete .rec-Dateiname abgeleitet.
        if file_path.endswith(".wav.not.mat"):
            return file_path.replace(".wav.not.mat", ".rec")
        if file_path.endswith(".not.mat"):
            return file_path.replace(".not.mat", ".rec")
        if file_path.endswith(".wav"):
            return file_path.replace(".wav", ".rec")
        return file_path + ".rec"

    if feedback_pattern is None:
        # Default-Pattern für Feedback-/Catch-Einträge in den rec-Dateien.
        # Erwartet werden vier Gruppen:
        #   1) Zeitstempel
        #   2) Typ (FB oder catch)
        #   3) Quell-/Dateiname
        #   4) Template-ID
        feedback_pattern = r"([\d\.]+E\+?\d+) msec: (FB|catch) # ([A-Za-z0-9_\.\\/:]+) : Templ = (\d+)"
    feedback_re = re.compile(feedback_pattern)

    with_feedback = []
    without_feedback = []
    missing_rec = []
    missing_pattern = []

    for file_path in sorted(labels_dict.keys()):
        rec_path = _rec_path_from_label_path(file_path)

        # Wenn die rec-Datei fehlt, kann für dieses Label-File natürlich keine
        # Feedback-Information bestimmt werden.
        if not os.path.exists(rec_path):
            missing_rec.append((file_path, rec_path))
            continue

        try:
            # Mit encoding="utf-8" und errors="ignore" bleibt das Parsing robust,
            # auch wenn einzelne rec-Dateien ungewöhnliche Zeichen enthalten.
            with open(rec_path, "r", encoding="utf-8", errors="ignore") as f:
                rec_text = f.read()
        except Exception as e:
            missing_rec.append((file_path, f"{rec_path} (read error: {e})"))
            continue

        # findall liefert eine Liste aller Treffer des Patterns. Ein File kann
        # also mehrere Feedback-/Catch-Ereignisse enthalten.
        matches = feedback_re.findall(rec_text)
        has_feedback = len(matches) > 0

        if has_feedback:
            with_feedback.append((file_path, rec_path, matches))
        else:
            without_feedback.append((file_path, rec_path, []))
            missing_pattern.append((file_path, rec_path))

    if print_per_file:
        # Die Ausgabe ist absichtlich gruppiert, damit man sofort sieht, welche
        # Dateien Treffer enthalten und welche nicht.
        print("\n=== Feedback-Matches: GEFUNDEN ===")
        if with_feedback:
            for file_path, _, matches in with_feedback:
                for m in matches:
                    time_ms, fb_type, src, templ = m
                    print(f"{file_path}: time={time_ms}, type={fb_type}, src={src}, templ={templ}")
        else:
            print("(keine)")

        print("\n=== Feedback-Matches: NICHT GEFUNDEN ===")
        if without_feedback:
            for file_path, _, _ in without_feedback:
                print(f"{file_path}: <kein Match>")
        else:
            print("(keine)")

        if missing_pattern:
            print("\n=== .rec ohne Treffer fuer feedback_pattern ===")
            for file_path, rec_path in missing_pattern:
                print(f"{file_path}: {rec_path}")

        if missing_rec:
            print("\n=== fehlende .rec-Dateien ===")
            for file_path, rec_path in missing_rec:
                print(f"{file_path}: {rec_path}")

        print(
            "\nSummary: "
            f"with_feedback={len(with_feedback)}, "
            f"without_feedback={len(without_feedback)}, "
            f"missing_pattern={len(missing_pattern)}, "
            f"missing_rec={len(missing_rec)}"
        )

    return {
        "with_feedback": with_feedback,
        "without_feedback": without_feedback,
        "missing_rec": missing_rec,
        "missing_pattern": missing_pattern,
    }

# creates a dictionary containing only files with feedback
def filter_labels_dict_training_feedback(
    labels_dict,
    feedback_summary=None,
    feedback_pattern=None,
    keep_other_phases=False,
    print_summary=True
    ):
    """
    Filtert labels_dict so, dass:
      - Baseline/Postbase komplett erhalten bleiben
      - Training nur Files mit Feedback-Treffer behaelt

        Diese Funktion ist der Brückenschritt zwischen Parsing und Plotting:
        summarize_feedback_information() sagt nur, welche rec-Dateien Feedback
        enthalten; hier wird diese Information benutzt, um labels_dict selbst auf
        die gewünschte Teilmenge zu reduzieren.

    Parameters
    ----------
    labels_dict : dict
        file_path -> label string
    feedback_summary : dict or None
        Optionales Ergebnis von summarize_feedback_information(...), damit die
        .rec-Dateien nicht erneut gelesen werden muessen.
    feedback_pattern : str or None
        Optionales Regex fuer summarize_feedback_information, wenn
        feedback_summary nicht uebergeben wird.
    keep_other_phases : bool
        Falls True, werden nicht-base/train Ordner unveraendert behalten.
    print_summary : bool
        Falls True, wird eine kurze Zusammenfassung gedruckt.

    Returns
    -------
    filtered_labels_dict : dict
    """
    if feedback_summary is None:
        # Wenn die Feedback-Zusammenfassung noch nicht existiert, wird sie hier
        # einmal intern berechnet. So kann die Funktion entweder direkt mit dem
        # Regex arbeiten oder ein bereits vorhandenes Ergebnis weiterverwenden.
        feedback_summary = summarize_feedback_information(
            labels_dict,
            print_per_file=False,
            feedback_pattern=feedback_pattern,
        )

    # Für die eigentliche Filterung reicht die Menge der Filepaths mit Feedback.
    feedback_files = {file_path for file_path, _, _ in feedback_summary["with_feedback"]}

    filtered = {}
    n_baseline_kept = 0
    n_training_kept = 0
    n_training_dropped = 0
    n_other_kept = 0

    for file_path, seq in labels_dict.items():
        # Die Phase wird ausschließlich über den Elternordner des Files bestimmt.
        # Das ist konsistent mit der restlichen Dateilogik im Skript.
        day = os.path.basename(os.path.dirname(file_path)).lower()
        is_baseline = day.startswith("base") or day.startswith("postbase")
        is_training = day.startswith("train")

        if is_baseline:
            # Baseline/Postbase bleibt immer vollständig erhalten.
            filtered[file_path] = seq
            n_baseline_kept += 1
            continue

        if is_training:
            # Bei Training wird selektiv gefiltert: nur Files mit nachgewiesenem
            # Feedback-Match bleiben im Dictionary.
            if file_path in feedback_files:
                filtered[file_path] = seq
                n_training_kept += 1
            else:
                n_training_dropped += 1
            continue

        # Alle sonstigen Phasen (falls vorhanden) können optional unverändert
        # mitgenommen werden.
        if keep_other_phases:
            filtered[file_path] = seq
            n_other_kept += 1

    if print_summary:
        print(
            "Filtered labels_dict: "
            f"total_in={len(labels_dict)}, total_out={len(filtered)}, "
            f"baseline_kept={n_baseline_kept}, "
            f"training_kept_with_feedback={n_training_kept}, "
            f"training_dropped_no_feedback={n_training_dropped}, "
            f"other_kept={n_other_kept}"
        )

    return filtered

# builds a color map for the raster plot, takes colors you give it
def build_syllable_color_map(
    syllables,
    fixed_colors=None,
    saturation=0.72,
    value=0.90,
):
    fixed_colors = {} if fixed_colors is None else dict(fixed_colors)
    syllables = list(syllables)

    # Reihenfolge erhalten und Duplikate entfernen.
    seen = set()
    ordered_syllables = []
    for s in syllables:
        if s not in seen:
            ordered_syllables.append(s)
            seen.add(s)

    def _to_hue(color):
        r, g, b = mcolors.to_rgb(color)
        h, _, _ = colorsys.rgb_to_hsv(r, g, b)
        return h

    def _circ_dist(a, b):
        d = abs(a - b)
        return min(d, 1.0 - d)

    color_map = {}
    used_hues = []

    for s in ordered_syllables:
        if s in fixed_colors:
            color_map[s] = fixed_colors[s]
            used_hues.append(_to_hue(fixed_colors[s]))

    remaining = [s for s in ordered_syllables if s not in color_map]
    if not remaining:
        return color_map

    # Dichte Kandidaten auf dem Hue-Wheel.
    candidate_hues = np.linspace(0.0, 1.0, 360, endpoint=False)

    for s in remaining:
        if not used_hues:
            best_h = 0.0
        else:
            best_h = None
            best_score = -1.0
            for h in candidate_hues:
                score = min(_circ_dist(h, uh) for uh in used_hues)
                if score > best_score:
                    best_score = score
                    best_h = float(h)

        rgb = colorsys.hsv_to_rgb(best_h, float(saturation), float(value))
        color_map[s] = mcolors.to_hex(rgb)
        used_hues.append(best_h)

    return color_map

# return the align position for the raster plot
def match_position_in_original(seq_orig, align_to_target, align_repeat_unit, align_repeat_index):
    if align_to_target is None:
        return None

    match = re.search(align_to_target, seq_orig)

    if match is None:
        return None

    # Align to a specific repeat unit inside the target match.
    # Example:
    # align_to_target="b+", align_repeat_unit="b", align_repeat_index=4
    # -> align to the 4th b.
    if align_repeat_unit is not None and align_repeat_index is not None:
        repeat_index = int(align_repeat_index)

        if repeat_index < 1:
            raise ValueError("align_repeat_index must be >= 1.")

        unit_len = len(align_repeat_unit)
        unit_offset = (repeat_index - 1) * unit_len

        if unit_offset >= len(match.group(0)):
            return None

        return match.start() + unit_offset

    # Default: align to target start.
    return match.start()

# get syllable number after target
def syllables_after_target(seq, pattern):
    if pattern is None:
        return np.nan
    match = re.search(pattern, seq)
    if match:
        return len(seq) - match.end()
    return np.nan

# summarize syllables for plotting (in this case meta-repeat 'dc')
def transform_seq_for_plot(seq):
    # Note: this is hard-coded for case 'dc' -> D
    plot_tokens = []
    orig_to_plot = [None] * len(seq)

    i = 0
    plot_idx = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == "d" and seq[i + 1] == "c":
            plot_tokens.append("D")
            orig_to_plot[i] = plot_idx
            orig_to_plot[i + 1] = plot_idx
            i += 2
            plot_idx += 1
        else:
            plot_tokens.append(seq[i])
            orig_to_plot[i] = plot_idx
            i += 1
            plot_idx += 1

    seq_plot = "".join(plot_tokens)
    return seq_plot, orig_to_plot

# translate sequence position to plot sequence position
def original_pos_to_plot_pos(orig_pos, orig_to_plot):
    if orig_pos is None:
        return None

    if len(orig_to_plot) == 0:
        return None

    idx = int(orig_pos)

    if idx < 0 or idx >= len(orig_to_plot):
        return None

    return float(orig_to_plot[idx])

# find alignment points in plot in each bout
def group_align_positions_plot(group_records, align_to_target, align_repeat_unit, align_repeat_index):
    positions = []
    for rec in group_records:
        orig_pos = match_position_in_original(rec["seq_orig"],align_to_target, align_repeat_unit,
                                                align_repeat_index)
        plot_pos = original_pos_to_plot_pos(orig_pos, rec["orig_to_plot"])
        positions.append(plot_pos)
    return positions

# build raster matrix for plot
def build_matrix(group_records, align_to_target, align_repeat_unit, align_repeat_index,
                  n_cols, left_pad, syllable_to_idx):
    group_align_positions = group_align_positions_plot(group_records, align_to_target,
                                                        align_repeat_unit, align_repeat_index)
    matrix = np.full((len(group_records), n_cols), np.nan)

    for row_idx, (rec, align_pos) in enumerate(zip(group_records, group_align_positions)):
        if align_to_target is not None and align_pos is not None:
            start_col = int(round(left_pad - align_pos))
        else:
            start_col = 0

        for col_offset, syllable in enumerate(rec["seq_plot"]):
            col_idx = start_col + col_offset
            if 0 <= col_idx < n_cols and syllable in syllable_to_idx:
                matrix[row_idx, col_idx] = syllable_to_idx[syllable]

    return matrix

# get number of syllables in a song
def extract_song_lengths(ldict, only_prefix=("train",)):
    if isinstance(only_prefix, str):
        only_prefix = (only_prefix,)
    only_prefix = tuple(prefix.lower() for prefix in only_prefix)

    lengths = []
    for file_path, labels in ldict.items():
        if not labels:
            continue

        day = os.path.basename(os.path.dirname(file_path)).lower()
        if not day.startswith(only_prefix):
            continue

        lengths.append(len(labels))

    return np.array(lengths, dtype=float)

def map_labels(label_series, combine_bk_chunks=False):
    """
    Map already-letter-coded CSV labels.

    Rule:
    - b after a currently open b-token -> k

    Optional:
    - b + k -> bk

    This mimics the old pre-pass behavior from the .not.mat pipeline.
    """
    symbols = list(label_series) if isinstance(label_series, str) else list(label_series)

    mapped = []

    for tok in symbols:
        if tok == "b" and mapped and mapped[-1] == "b":
            mapped.append("k")
        else:
            mapped.append(tok)

    if not combine_bk_chunks:
        return mapped

    result = []
    i = 0

    while i < len(mapped):
        if i < len(mapped) - 1 and mapped[i] == "b" and mapped[i + 1] == "k":
            result.append("bk")
            i += 2
        else:
            result.append(mapped[i])
            i += 1

    return result

def build_chunk_sequence(
    bout_sequence,
    label_merge_map=None,
    repeat_collapse_map=None,
    exclude_labels=None,
    combine_bk_chunks=True,
):
    if label_merge_map is None:
        label_merge_map = {}
    if repeat_collapse_map is None:
        repeat_collapse_map = {}
    if exclude_labels is None:
        exclude_labels = set()

    if combine_bk_chunks:
        source_sequence = map_labels(
            bout_sequence,
            combine_bk_chunks=True,
        )
    else:
        source_sequence = list(bout_sequence) if isinstance(bout_sequence, str) else list(bout_sequence)

    chunk_sequence = []
    current_chunk_label = None
    seen_in_chunk = set()

    for ch in source_sequence:
        label = label_merge_map.get(ch, ch)
        if label in exclude_labels:
            current_chunk_label = None
            seen_in_chunk = set()
            continue

        if label != current_chunk_label:
            chunk_sequence.append(label)
            current_chunk_label = label
            seen_in_chunk = {ch}
        else:
            if ch in seen_in_chunk:
                chunk_sequence.append(label)
                seen_in_chunk = {ch}
            else:
                seen_in_chunk.add(ch)

    if repeat_collapse_map:
        collapsed = []
        for lbl in chunk_sequence:
            new_lbl = repeat_collapse_map.get(lbl, lbl)
            if collapsed and collapsed[-1] == new_lbl:
                continue
            collapsed.append(new_lbl)
        chunk_sequence = collapsed

    return chunk_sequence

def compute_transition_matrix_and_labels_from_bouts(
    bout_list,
    exclude_labels=None,
    label_merge_map=None,
    repeat_collapse_map=None,
    combine_bk_chunks=True,
):
    """Berechnet Transitionen nur innerhalb einzelner Bouts (keine Bout-übergreifenden Übergänge)."""
    if exclude_labels is None:
        exclude_labels = set()
    if label_merge_map is None:
        label_merge_map = {}
    if repeat_collapse_map is None:
        repeat_collapse_map = {}

    unique_labels_set = set()
    for bout in bout_list:
        seq = build_chunk_sequence(
            bout,
            label_merge_map=label_merge_map,
            repeat_collapse_map=repeat_collapse_map,
            exclude_labels=exclude_labels,
            combine_bk_chunks=combine_bk_chunks,
        )
        unique_labels_set.update([lbl for lbl in seq if lbl not in exclude_labels])

    unique_labels = sorted(unique_labels_set)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    transition_matrix = np.zeros((len(unique_labels), len(unique_labels)))

    for bout in bout_list:
        seq = build_chunk_sequence(
            bout,
            label_merge_map=label_merge_map,
            repeat_collapse_map=repeat_collapse_map,
            exclude_labels=exclude_labels,
            combine_bk_chunks=combine_bk_chunks,
        )
        for i in range(len(seq) - 1):
            from_label = seq[i]
            to_label = seq[i + 1]
            if from_label in label_to_idx and to_label in label_to_idx:
                transition_matrix[label_to_idx[from_label], label_to_idx[to_label]] += 1

    return transition_matrix, unique_labels

def plot_transition_diagram(
    transition_matrix,
    labels,
    title="Transition Diagram",
    save_path=None,
    figsize=(10, 10),
    node_threshold=0,
    edge_threshold=1,
):
    """Plottet ein Transition Diagram basierend auf einer Transition Matrix."""

    node_counts = np.sum(transition_matrix, axis=0)
    total_count = np.sum(node_counts)

    if node_threshold > 0 and total_count > 0:
        keep_indices = np.where((node_counts / total_count * 100) > node_threshold)[0]
        transition_matrix = transition_matrix[np.ix_(keep_indices, keep_indices)]
        labels = [labels[i] for i in keep_indices]
        node_counts = np.sum(transition_matrix, axis=0)

    row_sums = np.sum(transition_matrix, axis=1)
    transition_probs = np.zeros_like(transition_matrix, dtype=float)
    for i in range(len(labels)):
        if row_sums[i] > 0:
            transition_probs[i] = transition_matrix[i] / row_sums[i]

    filtered_matrix = transition_matrix.copy()
    filtered_matrix[transition_probs * 100 < edge_threshold] = 0

    G = nx.from_numpy_array(filtered_matrix, create_using=nx.DiGraph)
    node_label_map = {idx: label for idx, label in enumerate(labels)}
    pos = nx.circular_layout(G)

    if np.max(node_counts) > 0:
        node_sizes = (node_counts / np.max(node_counts)) * 1500
    else:
        node_sizes = np.full(len(labels), 300)

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color="lightblue",
        alpha=1,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_label_map,
        font_size=10,
        ax=ax,
    )

    edges = list(G.edges())
    edge_weights = [filtered_matrix[u, v] for u, v in edges]

    if edge_weights:
        max_weight = max(edge_weights)
        edge_widths = [0.5 + (w / max_weight) * 5 for w in edge_weights]
    else:
        edge_widths = []

    nx.draw_networkx_edges(
        G,
        pos,
        node_size=node_sizes,
        width=edge_widths,
        edge_color="black",
        arrows=True,
        arrowsize=20,
        arrowstyle="->",
        connectionstyle="arc3,rad=0.1",
        ax=ax,
        alpha=1,
    )

    rad = 0.1
    t_label = 0.5

    for u, v in edges:
        weight = transition_matrix[u, v]
        if weight == 0:
            continue

        if u == v:
            continue

        prob = transition_probs[u, v]
        label = f"{int(prob * 100)}"

        p0 = np.asarray(pos[u], dtype=float)
        p2 = np.asarray(pos[v], dtype=float)

        d = p2 - p0
        dist = np.linalg.norm(d)
        if dist == 0:
            continue

        perp = np.array([-d[1], d[0]]) / dist
        p1 = (p0 + p2) / 2 + rad * dist * perp

        lx = (1 - t_label) ** 2 * p0[0] + 2 * (1 - t_label) * t_label * p1[0] + t_label ** 2 * p2[0]
        ly = (1 - t_label) ** 2 * p0[1] + 2 * (1 - t_label) * t_label * p1[1] + t_label ** 2 * p2[1]

        ax.text(
            lx,
            ly,
            label,
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7),
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig, ax

def print_transition_probabilities_after_thresholds(
    transition_matrix,
    labels,
    node_threshold=0,
    edge_threshold=1,
):
    matrix = transition_matrix.copy()
    labels = list(labels)

    # --- same node filter as plot_transition_diagram ---
    node_counts = np.sum(matrix, axis=0)
    total_count = np.sum(node_counts)

    if node_threshold > 0 and total_count > 0:
        keep_indices = np.where((node_counts / total_count * 100) > node_threshold)[0]
        matrix = matrix[np.ix_(keep_indices, keep_indices)]
        labels = [labels[i] for i in keep_indices]

    # --- transition probabilities after node filtering ---
    row_sums = np.sum(matrix, axis=1)
    probs = np.zeros_like(matrix, dtype=float)

    for i in range(len(labels)):
        if row_sums[i] > 0:
            probs[i] = matrix[i] / row_sums[i]

    print(
        f"\nTransition probabilities after thresholds "
        f"(node>{node_threshold}%, edge>{edge_threshold}%):"
    )

    has_edges = False

    for i, from_label in enumerate(labels):
        if row_sums[i] == 0:
            continue

        for j, to_label in enumerate(labels):
            count = matrix[i, j]
            prob = probs[i, j] * 100

            if count > 0 and prob >= edge_threshold:
                has_edges = True
                print(
                    f"{from_label} → {to_label}: "
                    f"{prob:.1f}%  (n={int(count)}/{int(row_sums[i])})"
                )

    if not has_edges:
        print("  [none]")

def clean_repeat_label(rep):
    label = str(rep)
    label = re.sub(r"\(\?:", "", label)
    label = re.sub(r"\{[^}]*\}", "", label)
    label = re.sub(r"[\(\)\[\]\{\}\?\:]", "", label)
    label = re.sub(r"[+*^$\\|.]", "", label)
    label = label.strip()
    if label == "jm": # special case for bird4
        return "j" 
    return label

def holm_correct_pvalues(pvalues_by_key):
    finite_items = [
        (key, float(pval))
        for key, pval in pvalues_by_key.items()
        if np.isfinite(pval)
    ]
    finite_items.sort(key=lambda item: item[1])

    corrected = {key: np.nan for key in pvalues_by_key}
    running_max = 0.0
    m = len(finite_items)

    for idx, (key, pval) in enumerate(finite_items):
        adj = min(1.0, (m - idx) * pval)
        running_max = max(running_max, adj)
        corrected[key] = running_max

    return corrected

def choose_two_group_test(x, y, mode="auto"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if mode == "auto":
        normal_x = (len(x) >= 3) and (stats.shapiro(x).pvalue > 0.05)
        normal_y = (len(y) >= 3) and (stats.shapiro(y).pvalue > 0.05)
        if normal_x and normal_y:
            stat_val, p = stats.ttest_ind(x, y, equal_var=False)
            return "welch_t", float(stat_val), float(p)
        stat_val, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        return "mannwhitney", float(stat_val), float(p)

    if mode == "ttest":
        stat_val, p = stats.ttest_ind(x, y, equal_var=False)
        return "welch_t", float(stat_val), float(p)

    if mode == "mannwhitney":
        stat_val, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        return "mannwhitney", float(stat_val), float(p)

    raise ValueError("training_test must be 'auto', 'ttest', or 'mannwhitney'")

def discrete_split_violin(ax, x_center, base, train, width, col_base, col_train, alpha=0.8, line_width=8):
    if len(base) == 0 or len(train) == 0:
        return

    y_min = int(min(base.min(), train.min()))
    y_max = int(max(base.max(), train.max()))
    ys = np.arange(y_min, y_max + 1)

    base_counts = np.array([(base == y).sum() for y in ys], dtype=float)
    train_counts = np.array([(train == y).sum() for y in ys], dtype=float)

    base_w = base_counts / base_counts.max() if base_counts.max() > 0 else base_counts
    train_w = train_counts / train_counts.max() if train_counts.max() > 0 else train_counts

    for y, bw in zip(ys, base_w):
        ax.hlines(y, x_center - bw * width, x_center, color=col_base, alpha=alpha, lw=line_width, zorder=1)
    for y, tw in zip(ys, train_w):
        ax.hlines(y, x_center, x_center + tw * width, color=col_train, alpha=alpha, lw=line_width, zorder=1)


def flatten_lengths(day_items):
    arrays = [vals["lengths"] for _, vals in day_items if len(vals["lengths"]) > 0]
    if len(arrays) == 0:
        return np.array([], dtype=float)
    return np.concatenate(arrays)

def pooled_rate(day_list):
    total_songs = 0
    total_hours = 0.0

    for _, vals in day_list:
        count = vals.get("count", 0)
        hours = vals.get("duration_hours", np.nan)

        if np.isnan(hours) or hours <= 0:
            continue

        total_songs += count
        total_hours += hours

    if total_hours <= 0:
        return np.nan, total_songs, total_hours

    return total_songs / total_hours, total_songs, total_hours


def flatten_repeats(day_items):
    arrays = [vals["repeats_per_song"] for _, vals in day_items if len(vals["repeats_per_song"]) > 0]
    if len(arrays) == 0:
        return np.array([], dtype=float)
    return np.concatenate(arrays)

def compute_percentage_df(df, categories):
    df = df.copy()
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()

    signal_total = df.loc["hits"] + df.loc["misses"]

    pct = pd.DataFrame(
        index=categories,
        columns=df.columns,
        dtype=float,
    )

    pct.loc["hits"] = df.loc["hits"] / signal_total * 100
    pct.loc["misses"] = df.loc["misses"] / signal_total * 100
    pct.loc["false hits"] = df.loc["false hits"] / df.loc["overall"] * 100

    return pct

def load_click_accuracy_percentages(
    file_path,
    categories,
    skiprows=3,
    ):
    xl = pd.ExcelFile(file_path)
    sheet_names = [
        sheet
        for sheet in xl.sheet_names
    ]

    all_pct = {}

    for sheet in sheet_names:
        df = pd.read_excel(
            file_path,
            sheet_name=sheet,
            skiprows=skiprows,
            index_col=0,
        )

        all_pct[sheet] = compute_percentage_df(df, categories)

    return all_pct

def make_click_accuracy_summary(all_pct, categories):
    rows = []
    for sheet, pct in all_pct.items():
        for category in categories:
            rows.append({
                "sheet": sheet,
                "category": category,
                "value": float(pct.loc[category].mean()),
            })

    return pd.DataFrame(rows)

def print_click_accuracy_summary(summary_df, categories):
    label = "All sheets"
    df = summary_df
    print(f"\n{label}:")
    for category in categories:
        vals = df.loc[
            df["category"] == category,
            "value"
        ].astype(float)
        mean_val = vals.mean()
        sem_val = sem_of_valid(vals)
        print(
            f"  {category}: "
            f"{mean_val:.2f} ± {sem_val:.2f} "
            f"(n = {len(vals)})"
        )