#!/usr/bin/env python3
"""
Relabel IKEA ASM ground truth from 33 action classes to 3 work-state classes:
  0 = idle          (NA, other — no meaningful assembly activity)
  1 = in_between    (pick up, lay down, flip, rotate, push, position — transitional)
  2 = busy          (align, attach, insert, slide, spin, tighten — core assembly)

Usage:
    python relabel_ground_truth.py \
        --src_gt_dir   ../mstcn_data/groundTruth/ \
        --dst_gt_dir   ./data/groundTruth/ \
        --dst_mapping  ./data/mapping.txt

This reads every .txt ground-truth file from the original 33-class data,
maps each frame label to the 3-class scheme, and writes the result.
It also generates a new mapping.txt for the 3-class setup.
"""

import os
import argparse
import shutil

#  Class mapping 
# Original action names → 3-class labels
# Based on Table 11 of the IKEA ASM paper (action IDs 0–32)

IDLE_ACTIONS = {
    "NA",           # ID 0 — no annotation
    "other",        # ID 17 — unavailable action class
}

TRANSITION_ACTIONS = {
    "flip shelf",                       # ID 6
    "flip table",                       # ID 7
    "flip table top",                   # ID 8
    "lay down back panel",              # ID 10
    "lay down bottom panel",            # ID 11
    "lay down front panel",             # ID 12
    "lay down leg",                     # ID 13
    "lay down shelf",                   # ID 14
    "lay down side panel",              # ID 15
    "lay down table top",               # ID 16
    "pick up back panel",               # ID 18
    "pick up bottom panel",             # ID 19
    "pick up front panel",              # ID 20
    "pick up leg",                      # ID 21
    "pick up pin",                      # ID 22
    "pick up shelf",                    # ID 23
    "pick up side panel",               # ID 24
    "pick up table top",                # ID 25
    "position the drawer right side up",# ID 26
    "push table",                       # ID 27
    "push table top",                   # ID 28
    "rotate table",                     # ID 29
}

BUSY_ACTIONS = {
    "align leg screw with table thread",                # ID 1
    "align side panel holes with front panel dowels",   # ID 2
    "attach drawer back panel",                         # ID 3
    "attach drawer side panel",                         # ID 4
    "attach shelf to table",                            # ID 5
    "insert drawer pin",                                # ID 9
    "slide bottom of drawer",                           # ID 30
    "spin leg",                                         # ID 31
    "tighten leg",                                      # ID 32
}

# New 3-class label names
STATE_MAP = {}
for a in IDLE_ACTIONS:
    STATE_MAP[a] = "idle"
for a in TRANSITION_ACTIONS:
    STATE_MAP[a] = "in_between"
for a in BUSY_ACTIONS:
    STATE_MAP[a] = "busy"

NEW_MAPPING = {
    "idle": 0,
    "in_between": 1,
    "busy": 2,
}


def relabel_file(src_path, dst_path):
    """Read a 33-class GT file, write a 3-class GT file."""
    with open(src_path, 'r') as f:
        lines = f.read().split('\n')

    # Remove trailing empty line if present
    if lines and lines[-1] == '':
        lines = lines[:-1]

    new_lines = []
    unmapped = set()
    for line in lines:
        label = line.strip()
        if label in STATE_MAP:
            new_lines.append(STATE_MAP[label])
        else:
            # Fallback: try to infer from verb
            unmapped.add(label)
            new_lines.append("idle")  # safe default

    with open(dst_path, 'w') as f:
        f.write('\n'.join(new_lines) + '\n')

    return unmapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_gt_dir', required=True,
                        help='Original 33-class groundTruth directory')
    parser.add_argument('--dst_gt_dir', required=True,
                        help='Output 3-class groundTruth directory')
    parser.add_argument('--dst_mapping', default=None,
                        help='Output mapping.txt path (default: <dst_gt_dir>/../mapping.txt)')
    args = parser.parse_args()

    os.makedirs(args.dst_gt_dir, exist_ok=True)

    # Write new mapping file
    mapping_path = args.dst_mapping or os.path.join(os.path.dirname(args.dst_gt_dir.rstrip('/')), 'mapping.txt')
    with open(mapping_path, 'w') as f:
        for name, idx in sorted(NEW_MAPPING.items(), key=lambda x: x[1]):
            f.write(f"{idx} {name}\n")
    print(f"Wrote mapping: {mapping_path}")
    for name, idx in sorted(NEW_MAPPING.items(), key=lambda x: x[1]):
        print(f"  {idx} = {name}")

    # Process all ground truth files
    gt_files = [f for f in os.listdir(args.src_gt_dir) if f.endswith('.txt')]
    print(f"\nRelabeling {len(gt_files)} ground truth files...")

    all_unmapped = set()
    class_counts = {"idle": 0, "in_between": 0, "busy": 0}

    for gt_file in sorted(gt_files):
        src = os.path.join(args.src_gt_dir, gt_file)
        dst = os.path.join(args.dst_gt_dir, gt_file)
        unmapped = relabel_file(src, dst)
        all_unmapped |= unmapped

        # Count frames per class
        with open(dst, 'r') as f:
            for line in f:
                label = line.strip()
                if label in class_counts:
                    class_counts[label] += 1

    if all_unmapped:
        print(f"\n⚠ Unmapped labels (defaulted to 'idle'): {all_unmapped}")

    total = sum(class_counts.values())
    print(f"\nFrame distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: NEW_MAPPING[x[0]]):
        pct = 100 * count / total if total > 0 else 0
        print(f"  {cls:12s}: {count:>8,d} frames ({pct:.1f}%)")
    print(f"  {'TOTAL':12s}: {total:>8,d} frames")
    print("\nDone!")


if __name__ == '__main__':
    main()
