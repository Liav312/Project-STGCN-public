#!/usr/bin/env python3
"""Fine-tune ST-GCN on SU-EMD using hard triplet loss."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.stgcn_backbone import load_pretrained


class WindowDataset(Dataset):
    def __init__(self, root: Path, window: int, stride: int) -> None:
        self.windows: List[np.ndarray] = []
        self.labels: List[int] = []
        for p in sorted(root.glob('*.npy')):
            angles = np.load(p)
            label = int(p.stem.split('A')[1][:2])  # action id
            for i in range(0, max(1, len(angles) - window + 1), stride):
                self.windows.append(angles[i:i + window].astype(np.float32))
                self.labels.append(label)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Return windows as (C, T) for ST-GCN
        return torch.from_numpy(self.windows[idx]).T, self.labels[idx]


class TripletDataset(Dataset):
    def __init__(self, base: WindowDataset) -> None:
        self.base = base
        self.triplets: List[Tuple[int, int, int]] = []

    def mine(self, model: torch.nn.Module, device: torch.device) -> None:
        loader = DataLoader(self.base, batch_size=64)
        feats: List[torch.Tensor] = []
        labels: List[int] = []
        with torch.no_grad():
            for x, y in loader:
                # x already in (B, C, T)
                x = x.to(device).unsqueeze(-1).unsqueeze(-1)
                z = model(x)
                feats.append(torch.nn.functional.normalize(z, dim=1).cpu())
                labels.extend(y.tolist())
        feats = torch.cat(feats)
        labels = np.array(labels)
        self.triplets.clear()
        for i in range(len(feats)):
            pos_mask = labels == labels[i]
            neg_mask = ~pos_mask
            pos_dists = 1 - (feats[i] @ feats[pos_mask].T)
            neg_dists = 1 - (feats[i] @ feats[neg_mask].T)
            if len(pos_dists) == 0 or len(neg_dists) == 0:
                continue
            j = pos_mask.nonzero()[pos_dists.argmax().item()][0]
            k = neg_mask.nonzero()[neg_dists.argmin().item()][0]
            self.triplets.append((i, j, k))

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, p, n = self.triplets[idx]
        return (torch.from_numpy(self.base.windows[a]).T,
                torch.from_numpy(self.base.windows[p]).T,
                torch.from_numpy(self.base.windows[n]).T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--angles_dir', default='data/angles/suemd-markless')
    ap.add_argument('--checkpoint', default='checkpoints/stgcn16_ntu_angles.pth')
    ap.add_argument('--out_dir', default='checkpoints')
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--window', type=int, default=50)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=16)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_pretrained(args.checkpoint, args.checkpoint, in_channels=16, num_class=120)

    base_ds = WindowDataset(Path(args.angles_dir), args.window, args.stride)
    triplet_ds = TripletDataset(base_ds)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.TripletMarginLoss(margin=0.2)

    for epoch in range(args.epochs):
        if epoch < 3:
            for b in list(model.st_gcn_networks)[:3]:
                for p in b.parameters():
                    p.requires_grad = False
        else:
            for b in list(model.st_gcn_networks)[:3]:
                for p in b.parameters():
                    p.requires_grad = True
        triplet_ds.mine(model, device)
        loader = DataLoader(triplet_ds, batch_size=args.batch_size, shuffle=True)
        model.train()
        for a, p, n in tqdm(loader, desc=f'epoch {epoch+1}'):
            a = a.to(device).unsqueeze(-1).unsqueeze(-1)
            p = p.to(device).unsqueeze(-1).unsqueeze(-1)
            n = n.to(device).unsqueeze(-1).unsqueeze(-1)    
            za = model(a)
            zp = model(p)
            zn = model(n)
            loss = criterion(za, zp, zn)
            loss.backward()
            opt.step()
            opt.zero_grad()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / 'stgcn_triplet_suemd.pth')


if __name__ == '__main__':
    main()
