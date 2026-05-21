"""
Preprocess IKEA ASM dataset for MS-TCN temporal action segmentation.

Converts OpenPose per-frame pose predictions + ground-truth action labels
into the format expected by MS-TCN
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict


# IKEA ASM action label mapping (from paper Table 11)
ACTION_LABELS = {
    0: "NA",
    1: "align leg screw with table thread",
    2: "align side panel holes with front panel dowels",
    3: "attach drawer back panel",
    4: "attach drawer side panel",
    5: "attach shelf to table",
    6: "flip shelf",
    7: "flip table",
    8: "flip table top",
    9: "insert drawer pin",
    10: "lay down back panel",
    11: "lay down bottom panel",
    12: "lay down front panel",
    13: "lay down leg",
    14: "lay down shelf",
    15: "lay down side panel",
    16: "lay down table top",
    17: "other",
    18: "pick up back panel",
    19: "pick up bottom panel",
    20: "pick up front panel",
    21: "pick up leg",
    22: "pick up pin",
    23: "pick up shelf",
    24: "pick up side panel",
    25: "pick up table top",
    26: "position the drawer right side up",
    27: "push table",
    28: "push table top",
    29: "rotate table",
    30: "slide bottom of drawer",
    31: "spin leg",
    32: "tighten leg",
}


def parse_openpose_json(filepath):
    """
    Parse an OpenPose keypoints JSON file.

    Returns:
        keypoints: np.array of shape (19, 3) -> [x, y, confidence] per joint
        If no person detected, returns zeros.

    OpenPose 19 keypoints (COCO-18 + neck):
        0: nose, 1: neck, 2: right shoulder, 3: right elbow, 4: right wrist,
        5: left shoulder, 6: left elbow, 7: left wrist, 8: mid hip,
        9: right hip, 10: right knee, 11: right ankle,
        12: left hip, 13: left knee, 14: left ankle,
        15: right eye, 16: left eye, 17: right ear, 18: left ear
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    if not data.get('people') or len(data['people']) == 0:
        return np.zeros((19, 3), dtype=np.float32)

    # Take the first (or most confident) person
    if len(data['people']) == 1:
        person = data['people'][0]
    else:
        # Multiple people detected so pick the one with highest mean confidence
        best_conf = -1
        best_person = data['people'][0]
        for p in data['people']:
            kps = np.array(p['pose_keypoints_2d']).reshape(-1, 3)
            mean_conf = kps[:, 2].mean()
            if mean_conf > best_conf:
                best_conf = mean_conf
                best_person = p
        person = best_person

    kps = np.array(person['pose_keypoints_2d'], dtype=np.float32).reshape(-1, 3)
    return kps


def extract_pose_features(pose_dir, num_frames, include_confidence=True, normalize=False):
    all_keypoints = []
    missing_count = 0

    for frame_idx in range(num_frames):
        filename = f"scan_video_{frame_idx:012d}_keypoints.json"
        filepath = os.path.join(pose_dir, filename)

        if os.path.exists(filepath):
            kps = parse_openpose_json(filepath)
        else:
            # Frame missing use zeros (will be handled by the model)
            kps = np.zeros((19, 3), dtype=np.float32)
            missing_count += 1

        all_keypoints.append(kps)

    if missing_count > 0:
        pct = missing_count / num_frames * 100
        if pct > 10:
            print(f"    WARNING: {missing_count}/{num_frames} frames missing ({pct:.1f}%)")

    all_keypoints = np.array(all_keypoints)  # (T, 19, 3)

    if normalize:
        # Normalize x,y to [0, 1] based on per-video min/max of non-zero joints
        for dim in [0, 1]:  # x, y
            vals = all_keypoints[:, :, dim]
            nonzero_mask = vals > 0
            if nonzero_mask.any():
                vmin = vals[nonzero_mask].min()
                vmax = vals[nonzero_mask].max()
                if vmax > vmin:
                    all_keypoints[:, :, dim] = np.where(
                        nonzero_mask,
                        (vals - vmin) / (vmax - vmin),
                        0
                    )

    if include_confidence:
        # Flatten to (T, 57): [x0, y0, c0, x1, y1, c1, etc]
        features = all_keypoints.reshape(num_frames, -1)  # (T, 57)
    else:
        # Only x, y coordinates: (T, 38)
        features = all_keypoints[:, :, :2].reshape(num_frames, -1)  # (T, 38)

    # MS-TCN expects (D, T)  features-first
    return features.T


def create_ground_truth_file(labels, label_map):
    """Convert integer label array to list of label name strings."""
    return [label_map[int(l)] for l in labels]


def main():
    parser = argparse.ArgumentParser(description="Preprocess IKEA ASM for MS-TCN")
    parser.add_argument('--ikea_root', type=str, required=True,
                        help='Path to IKEA ASM dataset root')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for MS-TCN formatted data')
    parser.add_argument('--camera', type=str, default='dev1',
                        choices=['dev1', 'dev2', 'dev3'],
                        help='Camera view: dev1=front, dev2=side, dev3=top')
    parser.add_argument('--pose_source', type=str, default='openpose',
                        choices=['openpose', 'keypoint_rcnn'],
                        help='Which pose estimator predictions to use')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize joint coordinates to [0,1]')
    parser.add_argument('--no_conf', action='store_true',
                        help='Exclude confidence values from features')
    parser.add_argument('--gt_action_path', type=str, default=None,
                        help='Path to gt_action.npy (default: {ikea_root}/gt_action.npy)')
    parser.add_argument('--gt_segments_path', type=str, default=None,
                        help='Path to gt_segments.json (default: {ikea_root}/gt_segments.json)')
    args = parser.parse_args()

    # Resolve paths
    ikea_root = Path(args.ikea_root)
    output_dir = Path(args.output_dir)
    gt_action_path = Path(args.gt_action_path) if args.gt_action_path else ikea_root / 'gt_action.npy'
    gt_segments_path = Path(args.gt_segments_path) if args.gt_segments_path else ikea_root / 'gt_segments.json'
    pose_root = ikea_root / 'annotations' / 'pose_tracking'
    include_conf = not args.no_conf

    print(f"IKEA ASM -> MS-TCN Preprocessing")
    print(f"  IKEA root:    {ikea_root}")
    print(f"  Output:       {output_dir}")
    print(f"  Camera:       {args.camera}")
    print(f"  Pose source:  {args.pose_source}")
    print(f"  Normalize:    {args.normalize}")
    print(f"  Include conf: {include_conf}")
    print()

    # Load ground truth
    print("Loading ground truth...")
    gt_data = np.load(str(gt_action_path), allow_pickle=True).item()
    scan_names = gt_data['scan_name']
    gt_labels = gt_data['gt_labels']

    with open(str(gt_segments_path), 'r') as f:
        segments_db = json.load(f)['database']

    print(f"  {len(scan_names)} videos loaded")

    # Create output directories
    features_dir = output_dir / 'features'
    gt_dir = output_dir / 'groundTruth'
    splits_dir = output_dir / 'splits'
    for d in [features_dir, gt_dir, splits_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Write mapping file
    mapping_path = output_dir / 'mapping.txt'
    with open(mapping_path, 'w') as f:
        for idx in sorted(ACTION_LABELS.keys()):
            f.write(f"{idx} {ACTION_LABELS[idx]}\n")
    print(f"  Mapping written to {mapping_path}")

    # Process each video
    train_videos = []
    test_videos = []
    skipped = []
    stats = defaultdict(int)

    for i, (scan_name, labels) in enumerate(zip(scan_names, gt_labels)):
        furniture_type, video_name = scan_name.split('/')
        num_frames = len(labels)

        # Build path to pose predictions
        pose_dir = (pose_root / furniture_type / video_name /
                    args.camera / 'predictions' / 'pose2d' / args.pose_source)

        if not pose_dir.exists():
            skipped.append(scan_name)
            stats['skipped_no_pose'] += 1
            continue

        # Count available pose files
        pose_files = list(pose_dir.glob('*.json'))
        if len(pose_files) == 0:
            skipped.append(scan_name)
            stats['skipped_empty'] += 1
            continue

        # Use scan_name as video ID (replace / with _)
        video_id = scan_name.replace('/', '_')

        print(f"  [{i+1:3d}/{len(scan_names)}] {scan_name} "
              f"({num_frames} frames, {len(pose_files)} pose files)")

        # Handle frame count mismatch
        effective_frames = min(num_frames, len(pose_files))
        if abs(num_frames - len(pose_files)) > 5:
            print(f"    WARNING: frame count mismatch: GT={num_frames}, pose={len(pose_files)}")
            stats['frame_mismatch'] += 1

        # Extract features
        features = extract_pose_features(
            str(pose_dir),
            effective_frames,
            include_confidence=include_conf,
            normalize=args.normalize
        )

        # Truncate labels to match
        truncated_labels = labels[:effective_frames]

        # Save features: (D, T) as .npy
        np.save(str(features_dir / f"{video_id}.npy"), features)

        # Save ground truth: one label name per line
        gt_strings = create_ground_truth_file(truncated_labels, ACTION_LABELS)
        with open(gt_dir / f"{video_id}.txt", 'w') as f:
            f.write('\n'.join(gt_strings) + '\n')

        # Track train/test split
        subset = segments_db[scan_name]['subset']
        if subset == 'training':
            train_videos.append(video_id)
        else:
            test_videos.append(video_id)

        stats['processed'] += 1

    # Write split files
    # MS-TCN bundle format
    with open(splits_dir / 'train.split.bundle', 'w') as f:
        for vid in sorted(train_videos):
            f.write(f"{vid}\n")

    with open(splits_dir / 'test.split.bundle', 'w') as f:
        for vid in sorted(test_videos):
            f.write(f"{vid}\n")

    # Summary 
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"  Processed:    {stats['processed']} videos")
    print(f"  Training:     {len(train_videos)} videos")
    print(f"  Testing:      {len(test_videos)} videos")
    print(f"  Skipped:      {len(skipped)} videos")
    if stats['frame_mismatch']:
        print(f"  Frame mismatches: {stats['frame_mismatch']}")
    feat_dim = 57 if include_conf else 38
    print(f"  Feature dim:  {feat_dim} ({19} joints × {'3 (x,y,conf)' if include_conf else '2 (x,y)'})")
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    features/          ({stats['processed']} .npy files, shape ({feat_dim}, T))")
    print(f"    groundTruth/       ({stats['processed']} .txt files)")
    print(f"    splits/            (train.split.bundle, test.split.bundle)")
    print(f"    mapping.txt        (33 action classes)")

    if skipped:
        print(f"\nSkipped videos:")
        for s in skipped[:10]:
            print(f"    {s}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")


if __name__ == '__main__':
    main()
