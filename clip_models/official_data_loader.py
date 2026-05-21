#!/usr/bin/env python3
"""
official_data_loader.py — Clip dataset matching the IKEA ASM HCN/ST-GCN
official setup.

The underlying feature files are still your existing (57, T) arrays
(19 joints × 3 channels).  This loader reshapes to (19, 3, T), drops the
confidence dim, optionally drops the mid-hip joint, and returns 4D.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


# Helpers

def load_mapping(mapping_file):
    actions_dict = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split(' ', 1)
            actions_dict[name] = int(idx)
    return actions_dict


def load_video_list(split_file):
    with open(split_file, 'r') as f:
        return [v.strip() for v in f.read().split('\n') if v.strip()]


def _majority(labels):
    return int(np.argmax(np.bincount(labels.astype(int))))



# Dataset

class ClipDataset(Dataset):
    """Sliding-window clip dataset producing official-shape tensors.

    Returns
    -------
    clip  : FloatTensor (2, window, V, 1)   — (C, T, V, M)
    label : LongTensor  scalar
    """

    def __init__(
        self,
        vid_list_file,
        features_path,
        gt_path,
        actions_dict,
        window=32,
        stride=16,
        sample_rate=1,
        num_joints=18,       # 18 to match openpose layout (drops mid-hip)
        in_channels=2,       # x, y only — no confidence
    ):
        self.features_path = features_path
        self.gt_path = gt_path
        self.actions_dict = actions_dict
        self.window = window
        self.stride = stride
        self.sample_rate = sample_rate
        self.num_joints = num_joints
        self.in_channels = in_channels

        self.clips = []
        self.labels = []
        self.clip_label_count = np.zeros(len(actions_dict), dtype=np.int64)

        vids = load_video_list(vid_list_file)
        skipped = 0
        for vid in vids:
            feat_file = os.path.join(features_path, vid + '.npy')
            gt_file = os.path.join(gt_path, vid + '.txt')
            if not os.path.exists(feat_file) or not os.path.exists(gt_file):
                skipped += 1
                continue

            features = np.load(feat_file)             # (57, T_orig)
            features = features[:, ::sample_rate]     # (57, T)
            T = features.shape[1]

            with open(gt_file, 'r') as f:
                content = [l.strip() for l in f.read().split('\n') if l.strip()]
            content = content[::sample_rate]

            T = min(T, len(content))
            features = features[:, :T]
            content = content[:T]

            gt = np.array([actions_dict[c] for c in content], dtype=np.int64)

            start = 0
            while start + window <= T:
                clip_feat = features[:, start:start + window]
                lbl = _majority(gt[start:start + window])
                self.clips.append(clip_feat)
                self.labels.append(lbl)
                self.clip_label_count[lbl] += 1
                start += stride

        if skipped:
            print(f"[ClipDataset] Skipped {skipped} videos (missing files).")
        print(f"[ClipDataset] {len(vids) - skipped} videos → {len(self.clips)} clips "
              f"(window={window}, stride={stride})")

    @property
    def num_classes(self):
        return len(self.actions_dict)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        feat = self.clips[idx]         # (57, W)
        label = self.labels[idx]

        # (57, W) → (19, 3, W): joint, channel, time
        feat = feat.reshape(19, 3, self.window)

        # Drop confidence channel if in_channels == 2
        feat = feat[:, :self.in_channels, :]              # (19, 2, W)

        # Drop mid-hip joint (index 18) if num_joints == 18
        if self.num_joints == 18:
            feat = feat[:18]                              # (18, 2, W)

        # Reorder to (C, T, V): (channels, time, joints)
        feat = feat.transpose(1, 2, 0)                    # (2, W, V)

        # Add person dim: (C, T, V) → (C, T, V, M=1)
        feat = feat[..., None]

        return torch.FloatTensor(feat), torch.tensor(label, dtype=torch.long)


# Weighted sampler for class imbalance (mirrors official setup)

def make_weighted_sampler(dataset):
    """Build a WeightedRandomSampler that draws clips with probability
    inversely proportional to their class frequency."""
    from torch.utils.data import WeightedRandomSampler

    class_counts = dataset.clip_label_count.astype(np.float64)
    class_weights = 1.0 / (class_counts + 1e-9)
    sample_weights = np.array([class_weights[lbl] for lbl in dataset.labels])
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    print(f"  Class counts: {class_counts.astype(int).tolist()}")
    return sampler


# Video-level dataset (per-video sliding-window for eval)

class VideoDataset(Dataset):
    """Returns clips covering one full video for eval, with start indices.

    Output per clip: same (2, W, V, 1) shape as ClipDataset.
    """

    def __init__(self, features, window, stride, num_joints=18, in_channels=2):
        """features: (57, T) numpy array"""
        self.window = window
        self.stride = stride
        self.num_joints = num_joints
        self.in_channels = in_channels
        T = features.shape[1]
        self.T = T

        pad = max(0, window - T)
        if pad:
            features = np.concatenate(
                [features, np.repeat(features[:, -1:], pad, axis=1)], axis=1)
        self.features = features

        self.starts = []
        s = 0
        while s + window <= self.features.shape[1]:
            self.starts.append(s)
            s += stride
        last = self.features.shape[1] - window
        if last >= 0 and (not self.starts or self.starts[-1] != last):
            self.starts.append(last)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        feat = self.features[:, s:s + self.window]    # (57, W)
        feat = feat.reshape(19, 3, self.window)
        feat = feat[:, :self.in_channels, :]
        if self.num_joints == 18:
            feat = feat[:18]
        feat = feat.transpose(1, 2, 0)[..., None]     # (C, T, V, M)
        return torch.FloatTensor(feat), s
