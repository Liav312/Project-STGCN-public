#!/usr/bin/env python3
"""Convert xyz skeleton files to 16-angle representation.

This script loads all ``.npy`` files from a raw directory, smooths the
trajectories with a Savitzky-Golay filter and converts them to the 16 hinge
angles defined in :mod:`utils.angle_features`. The resulting arrays are
saved in half precision (``float16``) to ``output_dir`` using the same
filenames as the source files.
"""
from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.config_loader import load_config, Config

import numpy as np
from scipy.signal import savgol_filter

from utils.angle_features import JOINT_LABELS, compute_angles_mp15


# ---------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------

def _process_file(args: tuple[Path, Path, int, int]):
    src, out_dir, window, order = args
    data = np.load(src)
    data = data[:, : len(JOINT_LABELS), :]
    # apply Savitzky-Golay smoothing on each joint dimension
    flat = data.reshape(data.shape[0], -1)
    flat = savgol_filter(flat, window_length=window, polyorder=order, axis=0)
    data = flat.reshape(data.shape)
    cos,sin = compute_angles_mp15(data)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / (src.stem)
    np.savez(dst, cos=cos, sin=sin)
    return src.name


def convert_directory(src_dir: Path, out_dir: Path, workers: int = 32,
                      window: int = 9, order: int = 3) -> None:
    files = sorted(p for p in src_dir.glob('*.npy'))
    args = [(p, out_dir, window, order) for p in files]
    with Pool(processes=workers) as pool:
        for name in pool.imap_unordered(_process_file, args):
            print(f"Converted {name}")


# ---------------------------------------------------------------------------

def main() -> None:
    cfg: Config = load_config()

    src_root = Path("data/raw")
    out_root = Path("data/angles")
    window = 9
    order = 3
    workers = 32

    for sub in ['ntu',  'suemd-markless']:
        s = src_root / sub
        if not s.exists():
            continue
        o = out_root / sub
        print(f'Converting {s} -> {o}')
        convert_directory(s, o, workers, window, order)


if __name__ == '__main__':
    main()
