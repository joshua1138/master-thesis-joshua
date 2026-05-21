#!/usr/bin/env python3
"""
train_clipmodel.py — Train HCN or ST-GCN on the 3-class IKEA ASM work-state task.

Usage
-----
# Train ST-GCN
python train_clipmodel.py --model stgcn --data_dir ./data --epochs 50

# Train HCN
python train_clipmodel.py --model hcn --data_dir ./data --epochs 50

# With class weights (recommended — addresses idle/in_between imbalance)
python train_clipmodel.py --model stgcn --data_dir ./data --epochs 50 --weighted_loss

The script saves:
  ./clip_models/{model}/epoch-{N}.pt   — checkpoints every save_every epochs
  ./clip_models/{model}/best.pt        — best validation loss checkpoint
  ./clip_models/{model}/train_log.csv  — per-epoch metrics
"""

import os
import csv
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import ClipDataset, load_mapping, load_video_list
from stgcn_model import STGCN
from hcn_model import HCN


# Args

def parse_args():
    p = argparse.ArgumentParser(description='Train HCN or ST-GCN on IKEA ASM 3-class task')
    p.add_argument('--model',       required=True, choices=['stgcn', 'hcn'])
    p.add_argument('--data_dir',    default='./data')
    p.add_argument('--out_dir',     default='./clip_models',
                   help='Root output directory; model-specific subdir is created inside')
    p.add_argument('--window',      type=int,   default=64,
                   help='Clip window length in frames (default 64 ≈ 2.7s at 24fps)')
    p.add_argument('--stride',      type=int,   default=16,
                   help='Sliding window stride (default 16)')
    p.add_argument('--batch_size',  type=int,   default=64)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--weight_decay',type=float, default=1e-4)
    p.add_argument('--epochs',      type=int,   default=50)
    p.add_argument('--save_every',  type=int,   default=10)
    p.add_argument('--sample_rate', type=int,   default=1)
    # On Windows DataLoader workers use spawn-style multiprocessing which is slow
    # and prone to hangs; the dataset is held in memory anyway so 0 is fine.
    p.add_argument('--num_workers', type=int,
                   default=0 if os.name == 'nt' else 4)
    p.add_argument('--dropout',     type=float, default=0.5)
    p.add_argument('--weighted_loss', action='store_true',
                   help='Use inverse-frequency class weights in cross-entropy loss')
    p.add_argument('--seed',        type=int,   default=42)
    return p.parse_args()


# Class weights from training label distribution

def compute_class_weights(dataset, num_classes):
    counts = np.zeros(num_classes)
    for _, label in dataset:
        counts[label.item()] += 1
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes   # normalise
    print(f"  Class counts: {counts.astype(int)}")
    print(f"  Class weights: {np.round(weights, 3)}")
    return torch.FloatTensor(weights)


# One epoch

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for clips, labels in loader:
            clips  = clips.to(device)    # (B, 3, W, 19)
            labels = labels.to(device)   # (B,)

            logits = model(clips)        # (B, num_classes)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


# Main

def main():
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Train {args.model.upper()} — 3-Class Work-State ===")
    print(f"  Device     : {device}")
    print(f"  Window     : {args.window}  Stride: {args.stride}")
    print(f"  Batch size : {args.batch_size}  LR: {args.lr}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Weighted   : {args.weighted_loss}\n")

    # Output dirs
    model_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(model_dir, exist_ok=True)

    # Data
    mapping_file  = os.path.join(args.data_dir, 'mapping.txt')
    train_split   = os.path.join(args.data_dir, 'splits', 'train.split.bundle')
    test_split    = os.path.join(args.data_dir, 'splits', 'test.split.bundle')
    features_path = os.path.join(args.data_dir, 'features') + '/'
    gt_path       = os.path.join(args.data_dir, 'groundTruth') + '/'

    actions_dict = load_mapping(mapping_file)
    num_classes  = len(actions_dict)
    idx_to_name  = {v: k for k, v in actions_dict.items()}
    print(f"  Classes    : {num_classes}  ({', '.join(idx_to_name[i] for i in range(num_classes))})")

    print("\nBuilding datasets...")
    train_ds = ClipDataset(train_split, features_path, gt_path, actions_dict,
                           window=args.window, stride=args.stride,
                           sample_rate=args.sample_rate)
    test_ds  = ClipDataset(test_split,  features_path, gt_path, actions_dict,
                           window=args.window, stride=args.stride,
                           sample_rate=args.sample_rate)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Model
    if args.model == 'stgcn':
        model = STGCN(num_classes=num_classes, dropout=args.dropout)
    else:
        model = HCN(num_classes=num_classes, dropout=args.dropout)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Parameters : {n_params:,}")

    # Loss
    if args.weighted_loss:
        print("\nComputing class weights from training set...")
        weights = compute_class_weights(train_ds, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimiser + scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # CSV log
    log_path = os.path.join(model_dir, 'train_log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr', 'time_s']
        )

    best_val_loss = float('inf')
    print(f"\n{'Ep':>4}  {'TrainLoss':>10}  {'TrainAcc':>9}  "
          f"{'ValLoss':>9}  {'ValAcc':>8}  {'LR':>8}  {'Time':>6}")
    print('-' * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                          optimizer, device, train=True)
        val_loss,   val_acc   = run_epoch(model, test_loader,  criterion,
                                          optimizer, device, train=False)
        scheduler.step()
        elapsed = time.time() - t0

        lr_now = scheduler.get_last_lr()[0]
        print(f"{epoch:>4}  {train_loss:>10.4f}  {train_acc*100:>8.2f}%  "
              f"{val_loss:>9.4f}  {val_acc*100:>7.2f}%  {lr_now:>8.6f}  {elapsed:>5.1f}s")

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, round(train_loss, 5), round(train_acc, 5),
                round(val_loss, 5),   round(val_acc, 5),
                round(lr_now, 7), round(elapsed, 1)
            ])

        # Save checkpoint
        if epoch % args.save_every == 0:
            ckpt = os.path.join(model_dir, f'epoch-{epoch}.pt')
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_loss': val_loss, 'val_acc': val_acc}, ckpt)
            print(f"       → saved checkpoint: {ckpt}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_loss': val_loss, 'val_acc': val_acc},
                       os.path.join(model_dir, 'best.pt'))

    # Always save final
    torch.save({'epoch': args.epochs, 'model_state': model.state_dict()},
               os.path.join(model_dir, f'epoch-{args.epochs}.pt'))

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Logs: {log_path}")
    print(f"Models: {model_dir}/")


if __name__ == '__main__':
    main()
