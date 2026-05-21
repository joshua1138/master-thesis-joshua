#!/usr/bin/env python3
"""
plot_training.py — Plot training curves from a CSV log.

Usage
-----
python plot_training.py --log ./official_clip_models/stgcn/train_log.csv
python plot_training.py --log ./models_3c_causal/train_log.csv --title "MS-TCN causal (3-class)"
"""

import argparse
import csv
import os
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--log', required=True, help='Path to train_log.csv')
    p.add_argument('--out', default=None,
                   help='Output PNG path; defaults to <log_dir>/training_curves.png')
    p.add_argument('--title', default=None,
                   help='Optional figure title (otherwise inferred from log path)')
    p.add_argument('--smooth', type=int, default=1,
                   help='Optional moving-average window in epochs (1 = no smoothing)')
    return p.parse_args()


def load_log(path):
    epochs, tl, ta, vl, va = [], [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row['epoch']))
                tl.append(float(row['train_loss']))
                ta.append(float(row['train_acc']))
                vl.append(float(row['val_loss']))
                va.append(float(row['val_acc']))
            except (KeyError, ValueError):
                continue
    return epochs, tl, ta, vl, va


def smooth(xs, k):
    if k <= 1:
        return xs
    import numpy as np
    xs = np.asarray(xs, dtype=float)
    kernel = np.ones(k) / k
    return np.convolve(xs, kernel, mode='same').tolist()


def main():
    args = parse_args()

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required.  Install with: pip install matplotlib")
        sys.exit(1)

    if not os.path.exists(args.log):
        print(f"Log file not found: {args.log}")
        sys.exit(1)

    epochs, tl, ta, vl, va = load_log(args.log)
    if not epochs:
        print(f"No data found in {args.log}")
        sys.exit(1)

    if args.smooth > 1:
        tl, ta, vl, va = smooth(tl, args.smooth), smooth(ta, args.smooth), \
                         smooth(vl, args.smooth), smooth(va, args.smooth)

    # Best epoch (lowest val loss)
    best_idx = int(min(range(len(vl)), key=lambda i: vl[i]))
    best_epoch = epochs[best_idx]
    best_val = vl[best_idx]

    title = args.title or os.path.basename(os.path.dirname(os.path.abspath(args.log)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss plot
    ax1.plot(epochs, tl, label='train',   linewidth=1.8, color='#1f77b4')
    ax1.plot(epochs, vl, label='val',     linewidth=1.8, color='#d62728')
    ax1.axvline(best_epoch, linestyle='--', linewidth=1, color='gray',
                alpha=0.7, label=f'best @ ep {best_epoch}')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_title('Loss')
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, [a*100 for a in ta], label='train', linewidth=1.8, color='#1f77b4')
    ax2.plot(epochs, [a*100 for a in va], label='val',   linewidth=1.8, color='#d62728')
    ax2.axvline(best_epoch, linestyle='--', linewidth=1, color='gray',
                alpha=0.7, label=f'best @ ep {best_epoch}')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy (%)')
    ax2.set_title('Accuracy')
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle(f'{title}  —  best val_loss={best_val:.4f} @ epoch {best_epoch}',
                 fontsize=11)
    fig.tight_layout()

    out_path = args.out or os.path.join(os.path.dirname(args.log), 'training_curves.png')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"Saved curves to: {out_path}")


if __name__ == '__main__':
    main()
