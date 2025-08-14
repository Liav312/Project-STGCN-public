
import argparse, numpy as np, mmengine.fileio as fileio
from pathlib import Path

def build_len(root, out_pkl):
    root = Path(root)
    mapping = {}
    for npy in root.rglob("*.npz"):
        n_frames = np.load(npy, mmap_mode="r")["cos"].shape[0]
        mapping[str(npy)] = int(n_frames)
    fileio.dump(mapping, out_pkl)
    print(f"Wrote {len(mapping):,} entries → {out_pkl}")



d = ['train', 'val', 'test']


for name in d:
    build_len(f'data/split/suemd-markless/{name}', f'data/split/suemd-markless/suemd_clip_lengths_{name}.pkl')

# for name in d:
#     build_len(f'data/split/ntu/{name}', f'data/split/ntu/ntu_clip_lengths_{name}.pkl')
