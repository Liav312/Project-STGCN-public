# make_window_index.py  ⋅ run once per dataset (NTU angles, SU-EMD, …)
import os, glob, pickle, numpy as np
from tqdm import tqdm

WIN     = 50    # window length  (change via CLI if you like)
STRIDE  = 1     # stride
root    = 'data/split/ntu/val'               # where *.npy live
outpkl  = 'data/split/ntu/ntu_val_windows.pkl'

records = []
for path in tqdm(sorted(glob.glob(os.path.join(root, '*.npz')))):
    arr = np.load(path, mmap_mode='r')    # just to know T
    T   = arr.shape[0]

    # -- (A) very short clip  → one record with i0 = -1
    if T < WIN:
        cls = int(os.path.basename(path)[17:20]) - 1   # NTU naming A001→0 …
        records.append({'frame_dir': path, 'i0': -1, 'label': cls})
        continue

    # -- (B) normal clip  → slide windows
    for i0 in range(0, T - WIN + 1, STRIDE):
        cls = int(os.path.basename(path)[17:20]) - 1
        records.append({'frame_dir': path, 'i0': i0, 'label': cls})

pickle.dump(records, open(outpkl, 'wb'))
print(f'Wrote {len(records)} windows → {outpkl}')
