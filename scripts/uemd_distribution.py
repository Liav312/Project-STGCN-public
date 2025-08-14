#!/usr/bin/env python3
"""Display basic statistics for the SU-EMD raw dataset."""
from __future__ import annotations

import re
from pathlib import Path

from collections import Counter, defaultdict

from utils.config_loader import load_config

PAT = re.compile(r"S(\d+)A(\d+)D(\d+)R(\d+)")


def parse_name(name: str):
    match = PAT.search(name)
    if not match:
        return None
    return tuple(map(int, match.groups()))


def gather_counts(root: Path):
    counts = {
        "action": Counter(),
        "subject": Counter(),
        "duration": Counter(),
        "repeat": Counter(),
    }
    # Nested mapping of action -> day -> repetition count
    reps_by_action_day = defaultdict(lambda: Counter())


    for sub in ("suemd-vicon", "suemd-markless"):
        for p in (root / sub).glob("*.npy"):
            vals = parse_name(p.stem)
            if vals is None:
                continue
            s, a, d, r = vals
            counts["action"][a] += 1
            counts["subject"][s] += 1
            counts["duration"][d] += 1
            counts["repeat"][r] += 1
            reps_by_action_day[a][d] += 1

    return counts, reps_by_action_day



def print_counts(counts: dict[str, Counter]) -> None:
    for key, counter in counts.items():
        print(f"{key.capitalize()} distribution:")
        for k in sorted(counter):
            print(f"  {k}: {counter[k]}")
        print()



def print_action_day_counts(reps: dict[int, Counter]) -> None:
    print("Repetition counts per action/duration:")
    for a in sorted(reps):
        print(f"Action {a}:")
        for d in sorted(reps[a]):
            print(f"  Duration {d}: {reps[a][d]}")
        print()


def main() -> None:
    cfg = load_config()
    counts, reps_by_action_day = gather_counts(cfg.RAW_DATA_DIR)
    print_counts(counts)
    print_action_day_counts(reps_by_action_day)



if __name__ == "__main__":
    main()
