import os
import numpy as np
from pathlib import Path
import argparse

def check_npy_shapes(directory: Path, window: int = 50, required_features: int = 16):
    errors = []
    counter = 0

    for npy_file in directory.rglob("*.npy"):
        try:
            data = np.load(npy_file)
            if data.ndim != 2:
                errors.append(f"{npy_file} has {data.ndim} dims, expected 2.")
            elif data.shape[0] < window or data.shape[1] != required_features:
                errors.append(f"{npy_file} shape {data.shape} < ({window}, {required_features})")
                counter += 1
        except Exception as e:
            errors.append(f"{npy_file} failed to load: {e}")

    if not errors:
        print("✅ All files have shape >= (window, 16)")
    else:
        print("❌ Found issues:")
        for err in errors:
            print(" -", err)
    print(f"Total files checked: {counter }")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check shape of all .npy files in a directory recursively")
    parser.add_argument("dir", type=str, help="Path to directory to scan")
    parser.add_argument("--window", type=int, default=50, help="Minimum number of frames")
    parser.add_argument("--features", type=int, default=16, help="Minimum number of features (columns)")
    args = parser.parse_args()

    check_npy_shapes(Path(args.dir), args.window, args.features)
