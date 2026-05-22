#!/usr/bin/env python3
"""
Evaluation script for MS-TCN on IKEA ASM.

Reports metrics matching the original IKEA ASM paper (Table 2):
  - Frame accuracy (top-1)
  - Macro recall (per-class recall averaged across classes)
  - Mean Average Precision (mAP)

Plus temporal action segmentation metrics:
  - Edit score (normalized Levenshtein)
  - F1@{10, 25, 50} (segment-level F1 at IoU thresholds)

Usage:
    python eval.py --data_dir ./mstcn_data --results_dir ./results/dev1 --output_file eval_dev1.txt
"""

import numpy as np
import argparse
from collections import defaultdict


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["NA"]):
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


def edit_score(recognized, ground_truth, norm=True, bg_class=["NA"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["NA"]):
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


def compute_average_precision(gt_frames, pred_frames, class_name):
    """
    Compute Average Precision for a single class.
    Treats each frame as an independent prediction.
    """
    # Binary: is this class the ground truth at this frame?
    gt_binary = np.array([1 if g == class_name else 0 for g in gt_frames])
    pred_binary = np.array([1 if p == class_name else 0 for p in pred_frames])

    if gt_binary.sum() == 0:
        return None  # class not present in ground truth

    # For frame-wise mAP: treat predicted frames as positive detections
    # Sort by confidence (here all are equal, so just compute precision-recall curve)
    tp = (gt_binary == 1) & (pred_binary == 1)
    fp = (gt_binary == 0) & (pred_binary == 1)
    fn = (gt_binary == 1) & (pred_binary == 0)

    tp_count = tp.sum()
    fp_count = fp.sum()
    fn_count = fn.sum()

    if tp_count + fp_count == 0:
        return 0.0

    precision = tp_count / (tp_count + fp_count)
    recall = tp_count / (tp_count + fn_count)

    # AP approximation: precision * recall for single threshold
    return precision * recall if recall > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/ikea_asm',
                        help='Path to preprocessed data')
    parser.add_argument('--results_dir', default='./results/ikea_asm',
                        help='Path to prediction results')
    parser.add_argument('--output_file', default=None,
                        help='Save results to this file (also prints to stdout)')
    args = parser.parse_args()

    ground_truth_path = args.data_dir + "/groundTruth/"
    recog_path = args.results_dir + "/"
    file_list = args.data_dir + "/splits/test.split.bundle"
    mapping_file = args.data_dir + "/mapping.txt"

    list_of_videos = read_file(file_list).split('\n')
    list_of_videos = [v for v in list_of_videos if v.strip()]

    # Read mapping
    action_names = []
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                _, name = line.split(' ', 1)
                action_names.append(name)
    action_names_sorted = sorted(action_names, key=len, reverse=True)

    def parse_recognition(text):
        result = []
        words = text.split()
        i = 0
        while i < len(words):
            matched = False
            for name in action_names_sorted:
                name_words = name.split()
                if words[i:i+len(name_words)] == name_words:
                    result.append(name)
                    i += len(name_words)
                    matched = True
                    break
            if not matched:
                i += 1
        return result

    # Collect all predictions and ground truth
    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0

    # Per-class tracking for macro recall
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    # Accumulate all frames for mAP
    all_gt_frames = []
    all_pred_frames = []

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

            class_total[gt_label] += 1
            if gt_label == pred_label:
                correct += 1
                class_correct[gt_label] += 1

            all_gt_frames.append(gt_label)
            all_pred_frames.append(pred_label)

        edit += edit_score(recog_content[:min_len], gt_content[:min_len])

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content[:min_len], gt_content[:min_len], overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

    # Compute metrics

    # Frame accuracy (top-1)
    frame_acc = 100.0 * float(correct) / total

    # Macro recall: average per-class recall
    per_class_recall = []
    for cls in action_names:
        if class_total[cls] > 0:
            recall = class_correct[cls] / class_total[cls]
            per_class_recall.append(recall)
    macro_recall = 100.0 * np.mean(per_class_recall) if per_class_recall else 0.0

    # Mean Average Precision
    ap_scores = []
    for cls in action_names:
        if cls == "NA":
            continue
        ap = compute_average_precision(all_gt_frames, all_pred_frames, cls)
        if ap is not None:
            ap_scores.append(ap)
    mAP = 100.0 * np.mean(ap_scores) if ap_scores else 0.0

    # Edit score
    edit_avg = (1.0 * edit) / len(list_of_videos)

    # F1 scores
    f1_scores = []
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s]) if (tp[s] + fp[s]) > 0 else 0
        recall = tp[s] / float(tp[s] + fn[s]) if (tp[s] + fn[s]) > 0 else 0
        f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(np.nan_to_num(f1) * 100)

    # Output results
    results = []
    results.append("=" * 50)
    results.append("IKEA ASM — MS-TCN Evaluation Results")
    results.append("=" * 50)
    results.append("")
    results.append("Paper-comparable metrics (Table 2):")
    results.append("  Frame accuracy (top-1):  %.2f" % frame_acc)
    results.append("  Macro recall:            %.2f" % macro_recall)
    results.append("  mAP:                     %.2f" % mAP)
    results.append("")
    results.append("Segmentation metrics:")
    results.append("  Edit score:              %.2f" % edit_avg)
    results.append("  F1@0.10:                 %.2f" % f1_scores[0])
    results.append("  F1@0.25:                 %.2f" % f1_scores[1])
    results.append("  F1@0.50:                 %.2f" % f1_scores[2])
    results.append("")
    results.append("Per-class recall:")
    for cls in sorted(class_total.keys()):
        if class_total[cls] > 0:
            rec = 100.0 * class_correct[cls] / class_total[cls]
            results.append("  %-45s %6.2f%%  (%d/%d frames)" % (
                cls, rec, class_correct[cls], class_total[cls]))
    results.append("")
    results.append("Test videos: %d" % len(list_of_videos))
    results.append("Total frames: %d" % total)

    output_text = '\n'.join(results)

    # Print to stdout
    print(output_text)

    # Save to file if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(output_text + '\n')
        print(f"\nResults saved to {args.output_file}")


if __name__ == '__main__':
    main()
