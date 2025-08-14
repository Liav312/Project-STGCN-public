#!/usr/bin/env python3
"""Preprocess SU-EMD clips with smoothing, clipping, scaling and windowing."""
from __future__ import annotations

from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from utils.config_loader import load_config
from utils import preprocessing as pp


def process_file(src: Path, dst_dir: Path, cfg) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    data = np.load(src)
    processed, scale = pp.preprocess_sequence(data, cfg)
    windows = pp.slide_windows(processed, cfg.WINDOW_SIZE, cfg.STRIDE_LENGTH)
    np.save(dst, windows.astype(np.float32))
    print(f"{src.name}: scale={scale:.3f} windows={len(windows)} -> {dst}")


def main() -> None:
    cfg = load_config()
    print("Loaded config values:")
    for k, v in cfg.__dict__.items():
        print(f"  {k}: {v}")

    raw_root = cfg.RAW_DATA_DIR
    proc_root = cfg.PROCESSED_DATA_DIR

    for sub in ["suemd-vicon", "suemd-markless"]:
        src_dir = raw_root / sub
        dst_dir = proc_root / sub
        for src in src_dir.glob("*.npy"):
            process_file(src, dst_dir, cfg)


if __name__ == "__main__":
    main()
