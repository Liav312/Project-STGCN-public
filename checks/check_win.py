import numpy as np
from pathlib import Path
import argparse

def count_windows_in_dir(directory: Path, window: int, stride: int):
    total_windows = 0
    file_windows = {}

    for npy_file in directory.rglob("*.npy"):
        try:
            arr = np.load(npy_file, mmap_mode="r")  # shape (T, 16)
            T = arr.shape[0]
            if T < window:
                count = 1  # use padding
            else:
                count = (T - window) // stride + 1
            file_windows[str(npy_file)] = count
            total_windows += count
        except Exception as e:
            print(f"⚠️ Error with {npy_file}: {e}")

    print(f"\n✅ Total windows across all files: {total_windows}")
    for file, count in file_windows.items():
        print(f"{file}: {count} window(s)")
    return total_windows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count sliding windows in .npy files")
    parser.add_argument("dir", type=str, help="Directory containing .npy files")
    parser.add_argument("--window", type=int, default=50, help="Window size (T)")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    args = parser.parse_args()

    t = count_windows_in_dir(Path(args.dir), args.window, args.stride)
    print(t)
