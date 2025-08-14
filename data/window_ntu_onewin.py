import os, glob, pickle, numpy as np
from tqdm import tqdm
import random

WIN     = 50    # window length
STRIDE  = 1     # unused in new logic, but kept for compatibility
win_num=4
def win(root, outpkl):
    records = []
    for path in tqdm(sorted(glob.glob(os.path.join(root, '*.npz')))):
        arr = np.load(path, mmap_mode='r')    # just to know T
        T   = arr["cos"].shape[0]
        cls = int(os.path.basename(path)[17:20]) - 1   # NTU naming A001→0 …

        # (A) Very short clip: just one record with i0 = -1
        if T < WIN:
            records.append({'frame_dir': path, 'i0': -1, 'label': cls})
            continue

        # (B) Normal clip: one random window only
        range_len = T - WIN + 1
        n_options = min(win_num, range_len)

        for i0 in (random.sample(range(range_len), n_options) if range_len > 0 else []):
            records.append({'frame_dir': path, 'i0': i0, 'label': cls})

    pickle.dump(records, open(outpkl, 'wb'))
    print(f'Wrote {len(records)} windows → {outpkl}')



d = ['train', 'val', 'test']
for name in d:
    win(f'data/split/ntu/{name}', f'data/split/ntu/ntu_{name}_onewin.pkl')
