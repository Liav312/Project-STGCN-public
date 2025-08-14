#!/usr/bin/env python3
from __future__ import annotations

import argparse
import yaml
from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from config import Lift
from windowing import rep_detect, windows, export


def process_file(path: Path, ex: Lift, writer: export.Writer) -> None:
    data = np.load(path)
    angle = rep_detect.primary_angle(data, ex)
    reps = rep_detect.segment_reps(angle)
    for start, end in reps:
        rep = data[start:end]
        for w in windows.slice_windows(rep):
            phase = windows.get_phase_vec()
            writer.add(w.astype(np.float32), phase)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--map_yaml", required=True)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    mapping = yaml.safe_load(Path(args.map_yaml).read_text())

    writer = export.Writer(args.out_dir)

    for fname, ex_name in mapping.items():
        ex = Lift(ex_name)
        p = raw_dir / fname
        process_file(p, ex, writer)

    writer.close()


if __name__ == "__main__":
    main()
