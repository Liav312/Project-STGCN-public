#!/usr/bin/env python3
"""Analyze diversity statistics for NTU and SU-EMD-markerless datasets."""

from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict, Counter
import pickle
import numpy as np
import matplotlib.pyplot as plt

WINDOW_SIZE = 50

# ---------------------------------------------------------------------------
# Utility parsers
# ---------------------------------------------------------------------------
NTU_PAT = re.compile(r"S(\d+)C(\d+)P(\d+)R(\d+)A(\d+)")
SUEMD_PAT = re.compile(r"S(\d+)A(\d+)D(\d+)R(\d+)")


def parse_ntu(name: str) -> tuple[int, int, int, int, int] | None:
    match = NTU_PAT.search(name)
    if not match:
        return None
    return tuple(map(int, match.groups()))


def parse_suemd(name: str) -> tuple[int, int, int, int] | None:
    match = SUEMD_PAT.search(name)
    if not match:
        return None
    return tuple(map(int, match.groups()))


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def sequence_length(path: Path) -> int:
    arr = np.load(path, mmap_mode="r")
    return arr.shape[0]


def count_split_files(split_dir: Path) -> dict[str, int]:
    counts = {}
    for sub in ["train", "val", "test"]:
        folder = split_dir / sub
        counts[sub] = len(list(folder.glob("*.npy")))
    return counts


def diversity_by_split(split_dir: Path, parser, idx: int) -> dict[str, int]:
    """Return unique label counts for each split."""

    result = {}
    for sub in ["train", "val", "test"]:
        labels = set()
        for p in (split_dir / sub).glob("*.npy"):
            fields = parser(p.stem)
            if fields is None:
                continue
            labels.add(fields[idx])
        result[sub] = len(labels)
    return result



def count_windows(split_root: Path, prefix: str, is_ntu =0) -> dict[str, int]:
    """Return the number of sliding windows for each split."""

    counts = {}
    for sub in ["train", "val", "test"]:
        if is_ntu == 0 :    
            pkl = split_root / f"{prefix}_{sub}_windows.pkl"
        else:
            pkl = split_root / f"{prefix}_{sub}_onewin.pkl"
        with open(pkl, "rb") as f:
            records = pickle.load(f)
        counts[sub] = len(records)
    return counts


def diversity_by_windows(
    split_root: Path, prefix: str, parser, idx: int, is_ntu =0
) -> dict[str, int]:
    """Return unique label counts for windows of each split."""

    result = {}
    for sub in ["train", "val", "test"]:
        if is_ntu == 0 :    
            pkl = split_root / f"{prefix}_{sub}_windows.pkl"
        else:
            pkl = split_root / f"{prefix}_{sub}_onewin.pkl"
        with open(pkl, "rb") as f:
            records = pickle.load(f)
        labels = set()
        for rec in records:
            fields = parser(Path(rec["frame_dir"]).stem)
            if fields is None:
                continue
            labels.add(fields[idx])
        result[sub] = len(labels)
    return result


def window_label_count_by_split(
    split_root: Path, prefix: str, parser, idx: int,is_ntu =0
) -> dict[str, Counter]:
    """Return window counts per label for each split."""

    results: dict[str, Counter] = {}
    for sub in ["train", "val", "test"]:
        if is_ntu == 0 :    
            pkl = split_root / f"{prefix}_{sub}_windows.pkl"
        else:
            pkl = split_root / f"{prefix}_{sub}_onewin.pkl"
     
        with open(pkl, "rb") as f:
            records = pickle.load(f)
        counts = Counter()
        for rec in records:
            fields = parser(Path(rec["frame_dir"]).stem)
            if fields is None:
                continue
            counts[fields[idx]] += 1
        results[sub] = counts
    return results


def plot_window_label_distribution(
    counts: dict[str, Counter], title: str, ylabel: str, max_labels: int = 15
) -> None:
    """Plot stacked bars showing window counts per label in each split."""

    splits = ["train", "val", "test"]
    total = Counter()
    for c in counts.values():
        total.update(c)
    labels = [lbl for lbl, _ in total.most_common(max_labels)]
    other_counts = {
        split: sum(c[l] for l in c if l not in labels) for split, c in counts.items()
    }
    if any(other_counts[s] > 0 for s in splits):
        labels.append("other")

    bottoms = [0] * len(splits)
    for lbl in labels:
        heights = []
        for split in splits:
            if lbl == "other":
                value = other_counts[split]
            else:
                value = counts[split].get(lbl, 0)
            heights.append(value)
        plt.bar(splits, heights, bottom=bottoms, label=str(lbl))
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    plt.title(title)
    plt.xlabel("Split")
    plt.ylabel(ylabel)
    if len(labels) <= 15:
        plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()



# ---------------------------------------------------------------------------
# NTU dataset analysis
# ---------------------------------------------------------------------------

def analyze_ntu() -> None:
    root = Path("data/angles/ntu")
    split_root = Path("data/split/ntu")

    lengths = []
    actions = Counter()
    subjects = set()
    short = 0

    for path in root.glob("*.npy"):
        fields = parse_ntu(path.stem)
        if fields is None:
            continue
        _, _, subj, _, action = fields
        subjects.add(subj)
        actions[action] += 1
        l = sequence_length(path)
        lengths.append(l)
        if l < WINDOW_SIZE:
            short += 1

    split_counts = count_split_files(split_root)
    split_action_div = diversity_by_split(split_root, parse_ntu, 4)

    window_counts = count_windows(split_root, "ntu",1)
    window_action_div = diversity_by_windows(split_root, "ntu", parse_ntu, 4,1)
    window_action_counts = window_label_count_by_split(split_root, "ntu", parse_ntu, 4,1)


    print("NTU Dataset:")
    print(f"  Total sequences: {len(lengths)}")
    print(f"  Unique action classes: {len(actions)}")
    print(f"  Unique subjects: {len(subjects)}")
    print(
        "  Split sizes - Train: {train}, Val: {val}, Test: {test}".format(**split_counts)
    )
    print(
        "  Exercise diversity - Train: {train}, Val: {val}, Test: {test}".format(**split_action_div)
    )
    print(

        "  Windows per split - Train: {train}, Val: {val}, Test: {test}".format(**window_counts)
    )
    print(
        "  Window exercise diversity - Train: {train}, Val: {val}, Test: {test}".format(**window_action_div)
    )
    print(

        f"  {short} sequences ({short / len(lengths) * 100:.2f}%) < {WINDOW_SIZE} frames"
    )
    print()

    # Plots
    plt.figure(figsize=(6, 4))
    plt.bar(split_counts.keys(), split_counts.values(), color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.title("NTU Dataset – Split Distribution")
    plt.xlabel("Split")
    plt.ylabel("Number of sequences")
    plt.tight_layout()

    # Window counts
    plt.figure(figsize=(6, 4))
    plt.bar(window_counts.keys(), window_counts.values(), color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.title("NTU Dataset – Window Counts")
    plt.xlabel("Split")
    plt.ylabel("Number of windows")
    plt.tight_layout()

    # Window exercise diversity
    plt.figure(figsize=(6, 4))
    plt.bar(window_action_div.keys(), window_action_div.values(), color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.title("NTU – Window Exercise Diversity")
    plt.xlabel("Split")
    plt.ylabel("Unique exercises")
    plt.tight_layout()

    # Window action counts by split
    plt.figure(figsize=(8, 4))
    plot_window_label_distribution(
        window_action_counts,
        "NTU – Window counts per action",
        "Number of windows",
        max_labels=10,
    )


    # Action distribution
    plt.figure(figsize=(10, 4))
    actions_sorted = dict(sorted(actions.items()))
    plt.bar(actions_sorted.keys(), actions_sorted.values(), color="#1f77b4")
    plt.title("NTU – Number of sequences per action class")
    plt.xlabel("Action class")
    plt.ylabel("Count")
    plt.xticks(list(actions_sorted.keys()), rotation=90)
    plt.tight_layout()

    # Sequence length histogram
    plt.figure(figsize=(6, 4))
    plt.hist(lengths, bins=40, color="#2ca02c")
    plt.title("NTU – Distribution of sequence lengths")
    plt.xlabel("Frames per sequence")
    plt.ylabel("Count")
    plt.tight_layout()


# ---------------------------------------------------------------------------
# SU-EMD dataset analysis
# ---------------------------------------------------------------------------

def analyze_suemd() -> None:
    root = Path("data/angles/suemd-markless")
    split_root = Path("data/split/suemd-markless")

    lengths = []
    actions = Counter()
    subjects = set()
    durations = Counter()
    class_duration = defaultdict(lambda: Counter())
    short = 0

    for path in root.glob("*.npy"):
        fields = parse_suemd(path.stem)
        if fields is None:
            continue
        subj, action, dur, _ = fields
        subjects.add(subj)
        actions[action] += 1
        durations[dur] += 1
        class_duration[action][dur] += 1
        l = sequence_length(path)
        lengths.append(l)
        if l < WINDOW_SIZE:
            short += 1

    split_counts = count_split_files(split_root)
    split_action_div = diversity_by_split(split_root, parse_suemd, 1)
    split_duration_div = diversity_by_split(split_root, parse_suemd, 2)

    window_counts = count_windows(split_root, "suemd")
    window_action_div = diversity_by_windows(split_root, "suemd", parse_suemd, 1)
    window_duration_div = diversity_by_windows(split_root, "suemd", parse_suemd, 2)
    window_action_counts = window_label_count_by_split(split_root, "suemd", parse_suemd, 1)
    window_duration_counts = window_label_count_by_split(split_root, "suemd", parse_suemd, 2)


    print("SU-EMD-markerless Dataset:")
    print(f"  Total sequences: {len(lengths)}")
    print(f"  Unique action classes: {len(actions)}")
    print(f"  Unique subjects: {len(subjects)}")
    print(f"  Unique duration categories: {len(durations)}")
    print(
        "  Split sizes - Train: {train}, Val: {val}, Test: {test}".format(**split_counts)
    )
    print(
        "  Exercise diversity - Train: {train}, Val: {val}, Test: {test}".format(**split_action_div)
    )
    print(
        "  Duration diversity - Train: {train}, Val: {val}, Test: {test}".format(**split_duration_div)
    )
    print(

        "  Windows per split - Train: {train}, Val: {val}, Test: {test}".format(**window_counts)
    )
    print(
        "  Window exercise diversity - Train: {train}, Val: {val}, Test: {test}".format(**window_action_div)
    )
    print(
        "  Window duration diversity - Train: {train}, Val: {val}, Test: {test}".format(**window_duration_div)
    )
    print(

        f"  {short} sequences ({short / len(lengths) * 100:.2f}%) < {WINDOW_SIZE} frames"
    )
    print()

    # Plot: split distribution
    plt.figure(figsize=(6, 4))
    plt.bar(split_counts.keys(), split_counts.values(), color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.title("SU-EMD – Split Distribution")
    plt.xlabel("Split")
    plt.ylabel("Number of sequences")
    plt.tight_layout()

    # Window counts
    plt.figure(figsize=(6, 4))
    plt.bar(window_counts.keys(), window_counts.values(), color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.title("SU-EMD – Window Counts")
    plt.xlabel("Split")
    plt.ylabel("Number of windows")
    plt.tight_layout()

    # Window exercise diversity
    plt.figure(figsize=(6, 4))
    plt.bar(window_action_div.keys(), window_action_div.values(), color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.title("SU-EMD – Window Exercise Diversity")
    plt.xlabel("Split")
    plt.ylabel("Unique exercises")
    plt.tight_layout()

    # Window counts per action
    plt.figure(figsize=(8, 4))
    plot_window_label_distribution(
        window_action_counts,
        "SU-EMD – Window counts per action",
        "Number of windows",
    )

    # Window duration diversity
    plt.figure(figsize=(6, 4))
    plt.bar(window_duration_div.keys(), window_duration_div.values(), color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.title("SU-EMD – Window Duration Diversity")
    plt.xlabel("Split")
    plt.ylabel("Unique durations")
    plt.tight_layout()

    # Window counts per duration
    plt.figure(figsize=(6, 4))
    plot_window_label_distribution(
        window_duration_counts,
        "SU-EMD – Window counts per duration",
        "Number of windows",
    )

    # Action distribution
    plt.figure(figsize=(6, 4))
    actions_sorted = dict(sorted(actions.items()))
    plt.bar(actions_sorted.keys(), actions_sorted.values(), color="#1f77b4")
    plt.title("SU-EMD – Sequences per action class")
    plt.xlabel("Action class")
    plt.ylabel("Count")
    plt.xticks(list(actions_sorted.keys()))
    plt.tight_layout()

    # Duration distribution
    plt.figure(figsize=(6, 4))
    durations_sorted = dict(sorted(durations.items()))
    plt.bar(durations_sorted.keys(), durations_sorted.values(), color="#ff7f0e")
    plt.title("SU-EMD – Sequences per duration category")
    plt.xlabel("Duration category")
    plt.ylabel("Count")
    plt.xticks(list(durations_sorted.keys()))
    plt.tight_layout()

    # Combined action-duration distribution
    plt.figure(figsize=(8, 5))
    action_ids = sorted(actions)
    duration_ids = sorted(durations)
    bar_width = 0.8 / len(duration_ids)
    for i, d in enumerate(duration_ids):
        counts = [class_duration[a][d] for a in action_ids]
        plt.bar(
            [x + i * bar_width for x in range(len(action_ids))],
            counts,
            width=bar_width,
            label=f"Dur {d}"
        )
    plt.title("SU-EMD – Distribution of durations for each action class")
    plt.xlabel("Action class")
    plt.ylabel("Count")
    plt.xticks([
        x + bar_width * (len(duration_ids) - 1) / 2 for x in range(len(action_ids))
    ], action_ids)
    plt.legend(title="Duration")
    plt.tight_layout()

    # Sequence length histogram
    plt.figure(figsize=(6, 4))
    plt.hist(lengths, bins=40, color="#2ca02c")
    plt.title("SU-EMD – Distribution of sequence lengths")
    plt.xlabel("Frames per sequence")
    plt.ylabel("Count")
    plt.tight_layout()


if __name__ == "__main__":
    #analyze_ntu()
    analyze_suemd()
    plt.show()
