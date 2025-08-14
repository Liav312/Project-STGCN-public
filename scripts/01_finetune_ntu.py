#!/usr/bin/env python3
"""Fine-tune ST-GCN on angle data from NTU-120."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
# from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from utils.stgcn_backbone import load_pretrained

class AngleDataset(Dataset):
    """
    NTU-angle windows without RAM blow-up.
    We store (file-path, start_idx, label) tuples; the actual (40–50, 16)
    window is read from disk on-the-fly via NumPy mem-mapping.
    """

    def __init__(
        self,
        root: Path,
        window: int,
        stride: int,
        split: str = "train",
        val_ratio: float = 0.15,
    ) -> None:
        self.window = window
        self.samples: List[Tuple[Path, int, int]] = []  # (path, i0, label)

        for p in sorted(root.glob("*.npy")):
            label = int(p.stem.split('A')[1][:3]) - 1 
            seq_len = np.load(p, mmap_mode="r").shape[0]

            if seq_len < window:
                # Mark with i0 = -1  → __getitem__ will pad edge frames
                self.samples.append((p, -1, label))
            else:
                for i0 in range(0, seq_len - window + 1, stride):
                    self.samples.append((p, i0, label))

        # simple subject-agnostic 90/10 split
        split_pt = int(len(self.samples) * (1 - val_ratio))
        self.samples = (
            self.samples[:split_pt] if split == "train" else self.samples[split_pt:]
        )

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, i0, label = self.samples[idx]
        arr = np.load(path, mmap_mode="r")          # (T, 16)

        # -------- pick / pad one window --------
        if i0 == -1:                                # short clip
            pad  = self.window - arr.shape[0]
            pre  = pad // 2;  post = pad - pre
            win  = np.pad(arr, ((pre, post), (0, 0)), mode="edge")
        else:
            win  = arr[i0 : i0 + self.window]       # (T, 16)

        # -------- (T,C) -> (C,T,V)  with V = 25 --------
        x = torch.from_numpy(win.astype(np.float32)).T  # (16, T)
        x = x.unsqueeze(-1).expand(-1, -1, 25)          # (16, T, 25)

        return x, label



def train_epoch(model: torch.nn.Module, loader: DataLoader,
                opt: torch.optim.Optimizer, scaler: GradScaler,
                device: torch.device) -> None:
    model.train()
    ce = torch.nn.CrossEntropyLoss()
    for x, y in tqdm(loader, desc='train'):
        # x is already (B, C, T); add dummy V and M dimensions
        x = x.to(device).unsqueeze(-1).unsqueeze(-1)
        #x = x.to(device).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        y = y.to(device)
        with autocast(device_type='cuda', enabled=device.type=='cuda'):                   # ← AMP context
            logits = model(x)
            loss   = ce(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()


def eval_epoch(model: torch.nn.Module, loader: DataLoader,
               device: torch.device) -> float:
    model.eval()
    correct = 0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            # x is (B, C, T); add dummy joint and person dimensions
            x = x.to(device).unsqueeze(-1).unsqueeze(-1)
            y = y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            count += len(y)
    return correct / max(1, count)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--angles_dir', default='data/angles/ntu')
    ap.add_argument('--checkpoint', default='STG-CN/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-3d.pth')
    ap.add_argument('--out_dir', default='checkpoints')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--window', type=int, default=50)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=128)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = GradScaler(enabled=device.type == 'cuda')
    init_ckpt = out_dir / 'stgcn16_patch_init.pth'
    graph_cfg = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
    if not init_ckpt.is_file():
        load_pretrained(args.checkpoint, init_ckpt, in_channels=16,
                        num_class=120, graph_args=graph_cfg)

    model = load_pretrained(init_ckpt, init_ckpt, in_channels=16,
                        num_class=120, graph_args=graph_cfg).to(device)

    train_ds = AngleDataset(Path(args.angles_dir), args.window, args.stride, 'train')
    val_ds = AngleDataset(Path(args.angles_dir), args.window, args.stride, 'val')
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True, num_workers=8)

    # AdamW with bias & BatchNorm exempt from weight-decay
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad is False:
            continue
        if 'bias' in n or 'bn' in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    opt = torch.optim.AdamW(
            [{'params': decay,     'weight_decay': 0.05},
             {'params': no_decay,  'weight_decay': 0.0}],
            lr=3e-4,                # good starting LR for batch-128 AMP
            betas=(0.9, 0.999))

    # optional cosine schedule over <epochs> cycles
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(args.epochs):
        if epoch == 0:
            # freeze first two blocks
            for b in list(model.st_gcn_networks)[:2]:
                for p in b.parameters():
                    p.requires_grad = False
        elif epoch == 1:
            for b in list(model.st_gcn_networks)[:2]:
                for p in b.parameters():
                    p.requires_grad = True
        loop = tqdm(train_dl, desc=f'epoch {epoch+1}/{args.epochs}', leave=False)
        train_epoch(model, loop, opt, scaler, device)
        sched.step()

        if (epoch % 2 == 1) or (epoch == args.epochs - 1):
            acc = eval_epoch(model, val_dl, device)
            print(f'Epoch {epoch+1}  val-acc = {acc:.3f}')
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(),
                           out_dir / 'stgcn16_ntu_angles.pth')
            # early-stopping disabled; always train for the full epoch budget



if __name__ == '__main__':
    main()
