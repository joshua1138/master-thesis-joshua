#!/usr/bin/env python3
"""
Evaluation script for MS-TCN — work-state recognition (3-class or 33-class).

Reports:
  - Frame accuracy (top-1)
  - Macro recall (average across classes)
  - Mean average precision (mAP) — read from <results_dir>/probs/<video>.npy
  - Per-class precision, recall, F1, AP
  - Macro F1
  - Confusion matrix
  - Edit score (normalized Levenshtein)
  - F1@{10, 25, 50} (segment-level overlap metrics)

Usage:
    python eval.py --data_dir ./data --results_dir ./results --output_file eval_results.txt
"""

import os
import numpy as np
import argparse
from collections import defaultdict


def read_file(path):
    with open(path, 'r') as f:
        return f.read()


def get_labels_start_end_time(frame_wise_labels, bg_class=None):
    if bg_class is None:
        bg_class = []
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], float)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i
    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)
    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]
    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=None):
    if bg_class is None:
        bg_class = []
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=None):
    if bg_class is None:
        bg_class = []
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0
    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        if len(y_label) == 0:
            fp += 1
            continue
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def average_precision(scores, labels):
    """One-vs-rest AP for a single class.
    `scores` : 1D array of confidence scores for the class
    `labels` : 1D array of 0/1 ground-truth labels for that class
    Returns AP in [0, 1].
    """
    # Sort by score descending
    order = np.argsort(-scores)
    labels_sorted = labels[order]

    n_pos = labels_sorted.sum()
    if n_pos == 0:
        return float('nan')

    tp_cum = np.cumsum(labels_sorted)
    fp_cum = np.cumsum(1 - labels_sorted)

    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    recall    = tp_cum / n_pos

    # Standard AP: sum of (recall[i] - recall[i-1]) * precision[i]
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    ap = np.sum((recall - recall_prev) * precision)
    return float(ap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data',
                        help='Path to data (with groundTruth/, splits/, mapping.txt)')
    parser.add_argument('--results_dir', default='./results',
                        help='Path to prediction results (with optional probs/ subdir)')
    parser.add_argument('--output_file', default=None,
                        help='Save results to this file')
    args = parser.parse_args()

    ground_truth_path = os.path.join(args.data_dir, "groundTruth") + os.sep
    recog_path = args.results_dir + os.sep
    probs_path = os.path.join(args.results_dir, "probs")
    file_list = os.path.join(args.data_dir, "splits", "test.split.bundle")
    mapping_file = os.path.join(args.data_dir, "mapping.txt")

    list_of_videos = read_file(file_list).split('\n')
    list_of_videos = [v for v in list_of_videos if v.strip()]

    # Read mapping (preserve original index order so it lines up with probs rows)
    actions_dict = {}
    idx_to_action = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split(' ', 1)
            actions_dict[name] = int(idx)
            idx_to_action[int(idx)] = name

    num_classes = len(actions_dict)
    action_names = [idx_to_action[i] for i in range(num_classes)]

    # Robust parser for label files (handles multi-word class names)
    action_names_sorted = sorted(action_names, key=len, reverse=True)

    def parse_recognition(text):
        result = []
        words = text.split()
        i = 0
        while i < len(words):
            matched = False
            for name in action_names_sorted:
                name_words = name.split()
                if words[i:i + len(name_words)] == name_words:
                    result.append(name)
                    i += len(name_words)
                    matched = True
                    break
            if not matched:
                i += 1
        return result

    # === Collect predictions, ground truth, and (optionally) scores ===
    overlap = [.1, .25, .5]
    tp_seg, fp_seg, fn_seg = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0

    confusion = defaultdict(lambda: defaultdict(int))
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)

    # For mAP: accumulate per-frame ground-truth class indices and per-frame score vectors
    use_probs = os.path.isdir(probs_path)
    if use_probs:
        all_gt_idx = []
        all_scores = []
        missing_probs = []
    else:
        print(f"NOTE: {probs_path} not found — mAP will be skipped.\n"
              f"      Rerun `main.py --action predict` to regenerate softmax scores.\n")

    for vid in list_of_videos:
        gt_file = ground_truth_path + vid + ".txt"
        gt_content = read_file(gt_file).split('\n')[0:-1]

        recog_file = recog_path + vid
        recog_lines = read_file(recog_file).split('\n')
        recog_content = parse_recognition(recog_lines[1])

        min_len = min(len(gt_content), len(recog_content))

        for i in range(min_len):
            total += 1
            gt_label = gt_content[i]
            pred_label = recog_content[i]

            confusion[gt_label][pred_label] += 1

            if gt_label == pred_label:
                correct += 1
                class_tp[gt_label] += 1
            else:
                class_fp[pred_label] += 1
                class_fn[gt_label] += 1

        edit += edit_score(recog_content[:min_len], gt_content[:min_len], bg_class=[])

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content[:min_len], gt_content[:min_len],
                                     overlap[s], bg_class=[])
            tp_seg[s] += tp1
            fp_seg[s] += fp1
            fn_seg[s] += fn1

        # Accumulate scores for mAP, if probs are available
        if use_probs:
            probs_file = os.path.join(probs_path, vid + ".npy")
            if not os.path.exists(probs_file):
                missing_probs.append(vid)
                continue
            probs = np.load(probs_file)        # (num_classes, T_probs)
            T_probs = probs.shape[1]
            T_use = min(min_len, T_probs)
            # Build per-frame ground-truth index vector
            gt_idx = np.array([actions_dict[gt_content[i]] for i in range(T_use)],
                              dtype=np.int64)
            scores = probs[:, :T_use].T        # (T_use, num_classes)
            all_gt_idx.append(gt_idx)
            all_scores.append(scores)

    # === Frame-level metrics ===
    frame_acc = 100.0 * float(correct) / total

    per_class_metrics = {}
    for cls in action_names:
        tp_c = class_tp[cls]
        fp_c = class_fp[cls]
        fn_c = class_fn[cls]
        precision = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0
        recall    = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = tp_c + fn_c
        per_class_metrics[cls] = {
            'precision': precision, 'recall': recall, 'f1': f1, 'support': support
        }

    macro_recall = np.mean([m['recall'] for m in per_class_metrics.values()])
    macro_f1     = np.mean([m['f1']     for m in per_class_metrics.values()])

    # === mAP ===
    per_class_ap = {cls: float('nan') for cls in action_names}
    mAP = float('nan')
    if use_probs and all_scores:
        if missing_probs:
            print(f"WARNING: {len(missing_probs)} videos missing prob files, "
                  f"excluded from mAP")
        gt_idx_all  = np.concatenate(all_gt_idx,  axis=0)        # (N,)
        scores_all  = np.concatenate(all_scores,  axis=0)        # (N, num_classes)
        aps = []
        for c, cls in enumerate(action_names):
            binary_labels = (gt_idx_all == c).astype(np.float32)
            ap = average_precision(scores_all[:, c], binary_labels)
            per_class_ap[cls] = ap
            if not np.isnan(ap):
                aps.append(ap)
        mAP = float(np.mean(aps)) if aps else float('nan')

    edit_avg = (1.0 * edit) / len(list_of_videos)

    f1_scores = []
    for s in range(len(overlap)):
        precision = tp_seg[s] / float(tp_seg[s] + fp_seg[s]) if (tp_seg[s] + fp_seg[s]) > 0 else 0
        recall    = tp_seg[s] / float(tp_seg[s] + fn_seg[s]) if (tp_seg[s] + fn_seg[s]) > 0 else 0
        f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(np.nan_to_num(f1) * 100)

    # === Output ===
    results = []
    results.append("=" * 70)
    results.append(f"MS-TCN Evaluation  ({num_classes}-class)")
    results.append("=" * 70)
    results.append("")
    results.append("Frame-level metrics:")
    results.append("  Frame accuracy:     %.2f%%" % frame_acc)
    results.append("  Macro recall:       %.2f%%" % (macro_recall * 100))
    results.append("  Macro F1:           %.2f%%" % (macro_f1 * 100))
    if not np.isnan(mAP):
        results.append("  mAP:                %.2f%%" % (mAP * 100))
    else:
        results.append("  mAP:                N/A  (no probs/ directory found)")
    results.append("")

    results.append("Per-class metrics:")
    results.append("  %-20s  %8s  %8s  %8s  %8s  %8s" % (
        "Class", "Prec", "Recall", "F1", "AP", "Support"))
    results.append("  " + "-" * 70)
    for cls in action_names:
        m = per_class_metrics[cls]
        ap = per_class_ap[cls]
        ap_str = "%6.2f%%" % (ap * 100) if not np.isnan(ap) else "   N/A"
        results.append("  %-20s  %7.2f%%  %7.2f%%  %7.2f%%  %s  %8d" % (
            cls, m['precision'] * 100, m['recall'] * 100, m['f1'] * 100,
            ap_str, m['support']))
    results.append("")

    # Print confusion matrix only when it fits on screen (<=10 classes)
    if num_classes <= 10:
        results.append("Confusion matrix (rows=GT, cols=Pred):")
        header = "  %-20s" % "" + "".join(["  %-10s" % c for c in action_names])
        results.append(header)
        for gt_cls in action_names:
            row = "  %-20s" % gt_cls
            for pred_cls in action_names:
                row += "  %-10d" % confusion[gt_cls][pred_cls]
            results.append(row)
        results.append("")
    else:
        results.append("Confusion matrix omitted (>10 classes).")
        results.append("")

    results.append("Segmentation metrics:")
    results.append("  Edit score:         %.2f" % edit_avg)
    results.append("  F1@0.10:            %.2f%%" % f1_scores[0])
    results.append("  F1@0.25:            %.2f%%" % f1_scores[1])
    results.append("  F1@0.50:            %.2f%%" % f1_scores[2])
    results.append("")
    results.append("Test videos: %d" % len(list_of_videos))
    results.append("Total frames: %d" % total)

    output_text = '\n'.join(results)
    print(output_text)

    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(output_text + '\n')
        print(f"\nResults saved to {args.output_file}")


if __name__ == '__main__':
    main()
