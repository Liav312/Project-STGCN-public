import os
import shutil
import re
import random
from pathlib import Path
from collections import defaultdict

# --- CONFIGURATION ---
MAIN_DIR = "data/angles/suemd-markless"          # Input directory
OUTPUT_DIR = "data/split/suemd-markless"  # Output directory
EXTENSIONS = {".npz"}
RATIOS = dict(train=0.7, val=0.2, test=0.1)
SEED = 42

# --- REGEX FOR A AND D ---
PATTERN = re.compile(r"A(\d+).*D(\d+)")  # Matches A1D3 from S2A1D3R4

# --- CREATE SPLIT DIRS ---
for split in RATIOS:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# --- GROUP FILES BY (D, A) ---
grouped_files = defaultdict(list)
for file in Path(MAIN_DIR).glob("*"):
    if file.suffix not in EXTENSIONS:
        continue
    match = PATTERN.search(file.stem)
    if not match:
        print(f"Skipping {file.name} — no match for A and D")
        continue
    a_id, d_id = match.group(1), match.group(2)
    key = (d_id, a_id)
    grouped_files[key].append(file)

# --- SPLIT EACH GROUP ---
random.seed(SEED)
split_files = defaultdict(list)
a_ids_per_split = defaultdict(set)
d_ids_per_split = defaultdict(set)

for (d_id, a_id), files in grouped_files.items():
    random.shuffle(files)
    n = len(files)
    n_train = int(n * RATIOS["train"])
    n_val = int(n * RATIOS["val"])
    n_test = n - n_train - n_val

    split_files['train'].extend(files[:n_train])
    split_files['val'].extend(files[n_train:n_train + n_val])
    split_files['test'].extend(files[n_train + n_val:])

    if n_train:
        a_ids_per_split['train'].add(a_id)
        d_ids_per_split['train'].add(d_id)
    if n_val:
        a_ids_per_split['val'].add(a_id)
        d_ids_per_split['val'].add(d_id)
    if n_test:
        a_ids_per_split['test'].add(a_id)
        d_ids_per_split['test'].add(d_id)

# --- COPY FILES ---
for split, files in split_files.items():
    for f in files:
        dst = os.path.join(OUTPUT_DIR, split, f.name)
        shutil.copy2(f, dst)

# --- REPORT ---
print("\n=== Unique A and D values per Split ===")
for split in ['train', 'val', 'test']:
    print(f"{split.upper():<5}: {len(a_ids_per_split[split]):>3} A-types, {len(d_ids_per_split[split]):>3} D-types")