#!/usr/bin/env python3
"""
eval_clipmodel.py — Evaluate a trained HCN or ST-GCN on the IKEA ASM 3-class task.

Converts clip-level predictions back to per-frame predictions, then computes
the same metrics as the MS-TCN eval script so results are directly comparable:
  - Frame accuracy (top-1)
  - Macro recall, macro F1
  - Mean Average Precision (mAP) — from per-frame, per-class soft scores
  - Per-class precision / recall / F1 / AP
  - Confusion matrix
  - Edit score (segmental edit distance)
  - F1@{10, 25, 50} overlap thresholds

Soft scores are produced by averaging softmax probabilities from all clips
that overlap each frame (clip-level model → per-frame ranking score).

Usage
-----
# Evaluate best checkpoint
python eval_clipmodel.py --model stgcn --data_dir ./data

# Evaluate a specific checkpoint
python eval_clipmodel.py --model hcn --data_dir ./data \\
    --checkpoint ./clip_models/hcn/epoch-50.pt

# Evaluate on training split
python eval_clipmodel.py --model stgcn --data_dir ./data --split train
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import load_mapping, load_video_list, VideoDataset
from stgcn_model import STGCN
from hcn_model import HCN

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, average_precision_score
)


# Segmental metrics (edit distance + F1@k overlap)


def _run_length_encode(seq):
    """Convert frame-level sequence to (label, length) pairs."""
    if len(seq) == 0:
        return []
    rle = []
    cur, cnt = seq[0], 1
    for s in seq[1:]:
        if s == cur:
            cnt += 1
        else:
            rle.append((cur, cnt))
            cur, cnt = s, 1
    rle.append((cur, cnt))
    return rle


def edit_distance(s1, s2):
    """Levenshtein distance between two segment label sequences."""
    n, m = len(s1), len(s2)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = dp[i-1, j-1] if s1[i-1] == s2[j-1] else \
                       1 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    return dp[n, m]


def edit_score(pred_seq, gt_seq):
    pred_rle = [lbl for lbl, _ in _run_length_encode(pred_seq)]
    gt_rle   = [lbl for lbl, _ in _run_length_encode(gt_seq)]
    if not pred_rle and not gt_rle:
        return 100.0
    return max(0, (1 - edit_distance(pred_rle, gt_rle) /
                   max(len(pred_rle), len(gt_rle)))) * 100


def f1_at_k(pred_seq, gt_seq, overlap=0.5):
    """F1 score based on segment-level IoU overlap threshold."""
    pred_rle = _run_length_encode(pred_seq)
    gt_rle   = _run_length_encode(gt_seq)

    tp, fp = 0, 0
    gt_used = [False] * len(gt_rle)

    # build start/end frame indices
    def to_intervals(rle):
        ivs, s = [], 0
        for lbl, ln in rle:
            ivs.append((lbl, s, s + ln - 1))
            s += ln
        return ivs

    pred_ivs = to_intervals(pred_rle)
    gt_ivs   = to_intervals(gt_rle)

    for p_lbl, p_s, p_e in pred_ivs:
        best_iou, best_j = 0, -1
        for j, (g_lbl, g_s, g_e) in enumerate(gt_ivs):
            if gt_used[j] or p_lbl != g_lbl:
                continue
            inter = max(0, min(p_e, g_e) - max(p_s, g_s) + 1)
            union = (p_e - p_s + 1) + (g_e - g_s + 1) - inter
            iou   = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= overlap:
            tp += 1
            gt_used[best_j] = True
        else:
            fp += 1

    fn = sum(1 for used in gt_used if not used)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9) * 100


# Per-video prediction: clips → per-frame predictions + per-frame soft scores

def predict_video(model, features, window, stride, device):
    """
    features : np.ndarray (57, T)

    Returns
    -------
    per_frame_pred  : np.ndarray (T,)              — argmax label per frame
    per_frame_score : np.ndarray (T, num_classes)  — averaged softmax scores per frame
                      (each frame's score is the mean softmax over all clips covering it)
    """
    T = features.shape[1]
    vid_ds = VideoDataset(features, window=window, stride=stride)
    loader = DataLoader(vid_ds, batch_size=32, shuffle=False)

    # Accumulators: sum of softmax probs and number of contributing clips per frame
    prob_sum   = None  # (T_pad, num_classes)
    cover_cnt  = None  # (T_pad,)
    num_classes = None

    model.eval()
    with torch.no_grad():
        for clips, starts in loader:
            clips = clips.to(device)            # (B, 3, W, 19)
            logits = model(clips)               # (B, num_classes)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()

            if prob_sum is None:
                num_classes = probs.shape[1]
                T_pad     = features.shape[1] + window
                prob_sum  = np.zeros((T_pad, num_classes), dtype=np.float32)
                cover_cnt = np.zeros((T_pad,),             dtype=np.float32)

            starts_np = starts.numpy()
            for i, s in enumerate(starts_np):
                # Each clip covers [s, s+window) — add its softmax to those frames
                prob_sum[s:s + window]  += probs[i]
                cover_cnt[s:s + window] += 1

    # Trim to original T
    prob_sum  = prob_sum[:T]
    cover_cnt = cover_cnt[:T]

    # Average softmax across overlapping clips → per-frame class score
    cover_cnt_safe = np.maximum(cover_cnt, 1.0)[:, None]
    per_frame_score = prob_sum / cover_cnt_safe       # (T, num_classes)

    per_frame_pred  = np.argmax(per_frame_score, axis=1)
    return per_frame_pred, per_frame_score


# Main evaluation

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',       required=True, choices=['stgcn', 'hcn'])
    p.add_argument('--data_dir',    default='./data')
    p.add_argument('--out_dir',     default='./clip_models')
    p.add_argument('--checkpoint',  default=None,
                   help='Path to .pt checkpoint; defaults to best.pt')
    p.add_argument('--split',       default='test', choices=['train', 'test'])
    p.add_argument('--window',      type=int, default=64)
    p.add_argument('--stride',      type=int, default=16)
    p.add_argument('--sample_rate', type=int, default=1)
    p.add_argument('--dropout',     type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    mapping_file  = os.path.join(args.data_dir, 'mapping.txt')
    split_file    = os.path.join(args.data_dir, 'splits',
                                 f'{"train" if args.split == "train" else "test"}.split.bundle')
    features_path = os.path.join(args.data_dir, 'features') + '/'
    gt_path       = os.path.join(args.data_dir, 'groundTruth') + '/'
    model_dir     = os.path.join(args.out_dir, args.model)

    actions_dict = load_mapping(mapping_file)
    num_classes  = len(actions_dict)
    idx_to_name  = {v: k for k, v in actions_dict.items()}
    class_names  = [idx_to_name[i] for i in range(num_classes)]

    # Load model
    if args.model == 'stgcn':
        model = STGCN(num_classes=num_classes, dropout=args.dropout)
    else:
        model = HCN(num_classes=num_classes, dropout=args.dropout)

    ckpt_path = args.checkpoint or os.path.join(model_dir, 'best.pt')
    print(f"\n=== Eval {args.model.upper()} — {args.split} split ===")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Device     : {device}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    print(f"  Loaded epoch {ckpt.get('epoch', '?')}  "
          f"(val_loss={ckpt.get('val_loss', float('nan')):.4f}  "
          f"val_acc={ckpt.get('val_acc', float('nan'))*100:.2f}%)")

    vids = load_video_list(split_file)
    print(f"  Videos     : {len(vids)}\n")

    all_pred, all_gt = [], []
    all_scores = []   # list of (T_i, num_classes) — concatenated at the end for mAP
    edit_scores, f1_10, f1_25, f1_50 = [], [], [], []

    for vid in vids:
        feat_file = os.path.join(features_path, vid + '.npy')
        gt_file   = os.path.join(gt_path, vid + '.txt')
        if not os.path.exists(feat_file) or not os.path.exists(gt_file):
            continue

        features = np.load(feat_file)[:, ::args.sample_rate]   # (57, T)
        with open(gt_file, 'r') as f:
            content = [l.strip() for l in f.read().split('\n') if l.strip()]
        content = content[::args.sample_rate]

        T = min(features.shape[1], len(content))
        features = features[:, :T]
        content  = content[:T]
        gt = np.array([actions_dict[c] for c in content], dtype=np.int64)

        pred, scores = predict_video(model, features, args.window, args.stride, device)
        pred   = pred[:T]
        scores = scores[:T]

        all_pred.extend(pred.tolist())
        all_gt.extend(gt.tolist())
        all_scores.append(scores)

        edit_scores.append(edit_score(pred, gt))
        f1_10.append(f1_at_k(pred, gt, 0.10))
        f1_25.append(f1_at_k(pred, gt, 0.25))
        f1_50.append(f1_at_k(pred, gt, 0.50))

    all_pred   = np.array(all_pred)
    all_gt     = np.array(all_gt)
    all_scores = np.concatenate(all_scores, axis=0)   # (N_frames, num_classes)

    #  Frame-level metrics 
    frame_acc   = accuracy_score(all_gt, all_pred) * 100
    macro_rec   = recall_score(all_gt, all_pred, average='macro', zero_division=0) * 100
    macro_f1    = f1_score(all_gt, all_pred, average='macro', zero_division=0) * 100

    #  Mean Average Precision (per-class AP averaged) 
    gt_onehot = np.eye(num_classes, dtype=np.int64)[all_gt]   # (N_frames, num_classes)
    per_class_ap = []
    for c in range(num_classes):
        if gt_onehot[:, c].sum() == 0:
            per_class_ap.append(float('nan'))    # class absent in this split
            continue
        ap = average_precision_score(gt_onehot[:, c], all_scores[:, c])
        per_class_ap.append(ap)
    mAP = np.nanmean(per_class_ap) * 100

    #  Segment metrics 
    edit_mean   = np.mean(edit_scores)
    f1_10_mean  = np.mean(f1_10)
    f1_25_mean  = np.mean(f1_25)
    f1_50_mean  = np.mean(f1_50)

    print("=" * 55)
    print(f"  Frame accuracy  : {frame_acc:.2f}%")
    print(f"  Macro recall    : {macro_rec:.2f}%")
    print(f"  Macro F1        : {macro_f1:.2f}%")
    print(f"  mAP             : {mAP:.2f}%")
    print(f"  Edit score      : {edit_mean:.2f}")
    print(f"  F1@10           : {f1_10_mean:.2f}%")
    print(f"  F1@25           : {f1_25_mean:.2f}%")
    print(f"  F1@50           : {f1_50_mean:.2f}%")
    print("=" * 55)

    # Per-class AP table
    print("\nPer-class Average Precision:")
    for c, name in enumerate(class_names):
        ap = per_class_ap[c]
        ap_str = f"{ap*100:6.2f}%" if not np.isnan(ap) else "   N/A"
        print(f"  {name:>12}  AP = {ap_str}")

    print(f"\nPer-class report:")
    print(classification_report(all_gt, all_pred, target_names=class_names,
                                 zero_division=0))

    print("Confusion matrix (rows=GT, cols=Pred):")
    cm = confusion_matrix(all_gt, all_pred)
    header = "         " + "  ".join(f"{n:>10}" for n in class_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:>8}" + "  ".join(f"{v:>10}" for v in row))

    #  Save summary 
    summary_path = os.path.join(model_dir, f'eval_{args.split}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Model       : {args.model}\n")
        f.write(f"Checkpoint  : {ckpt_path}\n")
        f.write(f"Split       : {args.split}\n")
        f.write(f"Window      : {args.window}  Stride: {args.stride}\n\n")
        f.write(f"Frame accuracy  : {frame_acc:.2f}%\n")
        f.write(f"Macro recall    : {macro_rec:.2f}%\n")
        f.write(f"Macro F1        : {macro_f1:.2f}%\n")
        f.write(f"mAP             : {mAP:.2f}%\n")
        f.write(f"Edit score      : {edit_mean:.2f}\n")
        f.write(f"F1@10           : {f1_10_mean:.2f}%\n")
        f.write(f"F1@25           : {f1_25_mean:.2f}%\n")
        f.write(f"F1@50           : {f1_50_mean:.2f}%\n\n")
        f.write("Per-class Average Precision:\n")
        for c, name in enumerate(class_names):
            ap = per_class_ap[c]
            ap_str = f"{ap*100:6.2f}%" if not np.isnan(ap) else "   N/A"
            f.write(f"  {name:>12}  AP = {ap_str}\n")
        f.write("\n")
        f.write(classification_report(all_gt, all_pred,
                                       target_names=class_names, zero_division=0))

    # Also save raw per-frame scores so you can recompute/plot later without
    # re-running the model.
    scores_path = os.path.join(model_dir, f'scores_{args.split}.npz')
    np.savez_compressed(scores_path,
                        scores=all_scores.astype(np.float32),
                        labels=all_gt.astype(np.int64),
                        preds=all_pred.astype(np.int64),
                        class_names=np.array(class_names))
    print(f"\nSummary saved to: {summary_path}")
    print(f"Scores  saved to: {scores_path}")


if __name__ == '__main__':
    main()
