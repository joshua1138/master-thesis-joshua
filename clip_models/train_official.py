#!/usr/bin/env python3
"""
train_official.py — Train HCN or ST-GCN using Ben-Shabat et al.'s exact setup.

Faithful replication of the IKEA ASM official pose-baseline training:
  - 5D input (N, C, T, V, M=1) with C=2 (x, y only)
  - WeightedRandomSampler for class balancing
  - Adam optimizer, lr=1e-4, weight_decay=1e-6
  - MultiStepLR with milestones at [1000, 2000, 3000, 4000]
  - Per-frame loss via linear interpolation of clip-level logits
  - Plain CrossEntropyLoss (NO class weighting — the sampler handles imbalance)

Adds two things on top of the official setup:
  - Early stopping based on validation loss patience (default 15)
  - CSV logging of per-epoch metrics for plotting convergence curves

Usage
-----
# ST-GCN, 3-class
python train_official.py --model stgcn --data_dir ./data --max_epochs 300

# HCN, 33-class (longer training expected)
python train_official.py --model hcn --data_dir "C:/path/to/mstcn_data" --max_epochs 500 \\
    --out_dir ./official_33c
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from official_data_loader import (
    ClipDataset, load_mapping, make_weighted_sampler,
)
from official_stgcn import Model as STGCN
from official_hcn import HCN


# Args

def parse_args():
    p = argparse.ArgumentParser(description='Official HCN/ST-GCN replication on IKEA ASM')
    p.add_argument('--model',         required=True, choices=['stgcn', 'hcn'])
    p.add_argument('--data_dir',      default='./data')
    p.add_argument('--out_dir',       default='./official_clip_models')
    p.add_argument('--window',        type=int, default=32,
                   help='Frames per clip — official default is 32. Must be divisible by 16 for HCN.')
    p.add_argument('--stride',        type=int, default=16)
    p.add_argument('--batch_size',    type=int, default=128)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--weight_decay',  type=float, default=1e-6)
    p.add_argument('--max_epochs',    type=int, default=500,
                   help='Hard cap on epochs; early stopping usually triggers first')
    p.add_argument('--patience',      type=int, default=15,
                   help='Early-stopping patience on val loss')
    p.add_argument('--min_epochs',    type=int, default=20,
                   help='Always train at least this many epochs before allowing early stop')
    p.add_argument('--milestones',    type=int, nargs='+', default=[1000, 2000, 3000, 4000],
                   help='LR drop epochs (matches official MultiStepLR)')
    p.add_argument('--gamma',         type=float, default=0.1,
                   help='LR multiplier at each milestone')
    p.add_argument('--save_every',    type=int, default=50)
    p.add_argument('--num_workers',   type=int,
                   default=0 if os.name == 'nt' else 6)
    p.add_argument('--sample_rate',   type=int, default=1)
    p.add_argument('--seed',          type=int, default=42)
    return p.parse_args()


# Per-epoch train / eval pass

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """One pass over the loader.  Per-frame loss via linear interpolation
    of clip-level logits, exactly as in the official training script."""
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for clips, labels in loader:
            clips = clips.to(device)
            labels = labels.to(device)

            logits = model(clips)                   # (N, num_class)

            # Replicate the official per-frame loss: upsample clip logits to T frames
            # and compare against a label vector repeated across time.
            T = clips.size(2)                       # window size
            per_frame_logits = F.interpolate(
                logits.unsqueeze(-1), T,            # (N, num_class, T)
                mode='linear', align_corners=True
            )

            # Repeat the clip label across all T frames → (N, T)
            labels_t = labels.unsqueeze(1).expand(-1, T)

            loss = criterion(per_frame_logits, labels_t)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)

            # Frame-level accuracy on the interpolated logits
            preds = per_frame_logits.argmax(dim=1)  # (N, T)
            correct += (preds == labels_t).sum().item()
            total += labels.size(0) * T

    return total_loss / max(1, len(loader.dataset)), correct / max(1, total)


# Main

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Train {args.model.upper()} (official replication) ===")
    print(f"  Device      : {device}")
    print(f"  Window      : {args.window}  Stride: {args.stride}")
    print(f"  Batch       : {args.batch_size}  LR: {args.lr}  WD: {args.weight_decay}")
    print(f"  Max epochs  : {args.max_epochs}  Patience: {args.patience}")
    print(f"  Milestones  : {args.milestones}  Gamma: {args.gamma}")

    mapping_file  = os.path.join(args.data_dir, 'mapping.txt')
    train_split   = os.path.join(args.data_dir, 'splits', 'train.split.bundle')
    test_split    = os.path.join(args.data_dir, 'splits', 'test.split.bundle')
    features_path = os.path.join(args.data_dir, 'features') + os.sep
    gt_path       = os.path.join(args.data_dir, 'groundTruth') + os.sep

    actions_dict = load_mapping(mapping_file)
    num_classes = len(actions_dict)
    idx_to_name = {v: k for k, v in actions_dict.items()}
    print(f"  Classes     : {num_classes}  "
          f"({', '.join(idx_to_name[i] for i in range(min(num_classes, 10)))}"
          f"{'...' if num_classes > 10 else ''})")

    print("\nBuilding datasets...")
    train_ds = ClipDataset(train_split, features_path, gt_path, actions_dict,
                           window=args.window, stride=args.stride,
                           sample_rate=args.sample_rate)
    test_ds  = ClipDataset(test_split,  features_path, gt_path, actions_dict,
                           window=args.window, stride=args.stride,
                           sample_rate=args.sample_rate)

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Model
    if args.model == 'stgcn':
        model = STGCN(
            in_channels=2, num_class=num_classes,
            graph_args={'layout': 'openpose', 'strategy': 'spatial'},
            edge_importance_weighting=True, dropout=0.5,
        )
    else:
        model = HCN(
            in_channel=2, num_joint=18, num_person=1, out_channel=64,
            window_size=args.window, num_class=num_classes,
        )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Parameters  : {n_params:,}")

    criterion = nn.CrossEntropyLoss()    # no class weights — sampler handles it
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=args.milestones,
                                               gamma=args.gamma)

    # CSV log
    log_path = os.path.join(out_dir, 'train_log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr', 'time_s']
        )

    print(f"\n{'Ep':>4}  {'TrainLoss':>10}  {'TrainAcc':>9}  "
          f"{'ValLoss':>9}  {'ValAcc':>8}  {'LR':>10}  {'Time':>6}  Status")
    print('-' * 85)

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                          optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, test_loader, criterion,
                                       optimizer, device, train=False)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        # Track best
        improved = val_loss < best_val_loss
        status = ""
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_loss': val_loss, 'val_acc': val_acc},
                       os.path.join(out_dir, 'best.pt'))
            status = "★ best"
        else:
            epochs_no_improve += 1

        print(f"{epoch:>4}  {train_loss:>10.4f}  {train_acc*100:>8.2f}%  "
              f"{val_loss:>9.4f}  {val_acc*100:>7.2f}%  {lr_now:>10.6f}  "
              f"{elapsed:>5.1f}s  {status}")

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, round(train_loss, 5), round(train_acc, 5),
                round(val_loss, 5), round(val_acc, 5),
                round(lr_now, 8), round(elapsed, 1)
            ])

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_loss': val_loss, 'val_acc': val_acc},
                       os.path.join(out_dir, f'epoch-{epoch}.pt'))

        # Early stopping
        if epoch >= args.min_epochs and epochs_no_improve >= args.patience:
            print(f"\nEarly stopping — no val_loss improvement for "
                  f"{args.patience} epochs.")
            print(f"Best epoch: {best_epoch}  (val_loss={best_val_loss:.4f})")
            break

    print(f"\nTraining complete.  Best epoch: {best_epoch}  "
          f"(val_loss={best_val_loss:.4f})")
    print(f"Logs:   {log_path}")
    print(f"Models: {out_dir}/")
    print(f"\nNext steps:")
    print(f"  - Plot curves:  python plot_training.py --log {log_path}")
    print(f"  - Evaluate:     python eval_official.py --model {args.model} "
          f"--data_dir {args.data_dir} --out_dir {args.out_dir} --window {args.window}")


if __name__ == '__main__':
    main()
