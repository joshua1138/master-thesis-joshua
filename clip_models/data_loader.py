#!/usr/bin/env python3
"""
data_loader.py — Sliding-window clip dataset for HCN and ST-GCN.

Input feature files: (57, T)  — 19 joints × 3 (x, y, conf), T frames
Ground truth files : one action name per line (idle / in_between / busy)
Mapping file       : "0 idle\n1 in_between\n2 busy"

Output per clip
  ST-GCN : tensor (3, W, 19)  — channels-first (x/y/conf × time × joints)
  HCN    : tensor (W, 19, 3)  — (time × joints × channels)  [built in model]
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def load_mapping(mapping_file: str) -> dict:
    """Return {action_name: class_index}."""
    actions_dict = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, name = line.split(' ', 1)
            actions_dict[name] = int(idx)
    return actions_dict


def load_video_list(split_file: str) -> list:
    with open(split_file, 'r') as f:
        vids = [v.strip() for v in f.read().split('\n') if v.strip()]
    return vids


def _majority(labels: np.ndarray) -> int:
    counts = np.bincount(labels.astype(int))
    return int(np.argmax(counts))


# Dataset


class ClipDataset(Dataset):
    """
    Slides a window of length `window` with step `stride` over every video.

    Returns
    -------
    clip   : FloatTensor  (3, window, 19)  — ready for ST-GCN
             HCN model will permute internally to (window, 19, 3)
    label  : LongTensor   scalar — majority class in the window
    """

    def __init__(
        self,
        vid_list_file: str,
        features_path: str,
        gt_path: str,
        actions_dict: dict,
        window: int = 64,
        stride: int = 16,
        sample_rate: int = 1,
    ):
        self.features_path = features_path
        self.gt_path = gt_path
        self.actions_dict = actions_dict
        self.window = window
        self.stride = stride
        self.sample_rate = sample_rate

        self.clips = []   # list of (vid_name, start_frame)
        self.labels = []  # majority class for each clip

        vids = load_video_list(vid_list_file)
        skipped = 0
        for vid in vids:
            feat_file = os.path.join(features_path, vid + '.npy')
            gt_file   = os.path.join(gt_path,       vid + '.txt')

            if not os.path.exists(feat_file) or not os.path.exists(gt_file):
                skipped += 1
                continue

            features = np.load(feat_file)          # (57, T_orig)
            features = features[:, ::sample_rate]  # (57, T)
            T = features.shape[1]

            with open(gt_file, 'r') as f:
                content = [l.strip() for l in f.read().split('\n') if l.strip()]
            content = content[::sample_rate]

            # align lengths
            T = min(T, len(content))
            features = features[:, :T]
            content  = content[:T]

            gt = np.array([actions_dict[c] for c in content], dtype=np.int64)

            # slide window
            start = 0
            while start + window <= T:
                self.clips.append((vid, start, features[:, start:start+window]))
                self.labels.append(_majority(gt[start:start+window]))
                start += stride

        if skipped:
            print(f"[ClipDataset] Skipped {skipped} videos (missing files).")
        print(f"[ClipDataset] {len(vids) - skipped} videos → {len(self.clips)} clips "
              f"(window={window}, stride={stride})")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        vid, start, feat = self.clips[idx]   # feat: (57, W)
        label = self.labels[idx]

        # Reshape (57, W) → (3, W, 19): channels=(x,y,conf), time, joints
        # Feature layout: joint 0 → dims [0,1,2], joint 1 → [3,4,5], ...
        feat = feat.reshape(19, 3, self.window)   # (19, 3, W)
        feat = feat.transpose(1, 2, 0)            # (3, W, 19)
        feat = torch.FloatTensor(feat)

        return feat, torch.tensor(label, dtype=torch.long)


# Per-frame prediction loader


class VideoDataset(Dataset):
    """
    Returns one clip per window for a single video.
    Used at eval time to recover per-frame predictions.

    For frames not covered by a complete window (tail), the last window
    is extended/padded with the last valid frame.
    """

    def __init__(self, features: np.ndarray, window: int, stride: int):
        """
        features : (57, T)
        """
        self.window  = window
        self.stride  = stride
        T = features.shape[1]
        self.T = T

        # pad if needed so every frame is covered
        pad = max(0, window - T)
        if pad:
            features = np.concatenate(
                [features, np.repeat(features[:, -1:], pad, axis=1)], axis=1
            )
        self.features = features   # (57, T_padded)

        self.starts = []
        s = 0
        while s + window <= self.features.shape[1]:
            self.starts.append(s)
            s += stride
        # always include a window ending at the very last frame
        last = self.features.shape[1] - window
        if last >= 0 and (not self.starts or self.starts[-1] != last):
            self.starts.append(last)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        feat = self.features[:, s:s + self.window]   # (57, W)
        feat = feat.reshape(19, 3, self.window)
        feat = feat.transpose(1, 2, 0)               # (3, W, 19)
        return torch.FloatTensor(feat), s            # also return start index



def make_loaders(
    data_dir: str,
    window: int = 64,
    stride: int = 16,
    batch_size: int = 64,
    num_workers: int = 4,
    sample_rate: int = 1,
):
    mapping_file    = os.path.join(data_dir, 'mapping.txt')
    train_split     = os.path.join(data_dir, 'splits', 'train.split.bundle')
    test_split      = os.path.join(data_dir, 'splits', 'test.split.bundle')
    features_path   = os.path.join(data_dir, 'features') + '/'
    gt_path         = os.path.join(data_dir, 'groundTruth') + '/'

    actions_dict = load_mapping(mapping_file)

    train_ds = ClipDataset(train_split, features_path, gt_path, actions_dict,
                           window=window, stride=stride, sample_rate=sample_rate)
    test_ds  = ClipDataset(test_split,  features_path, gt_path, actions_dict,
                           window=window, stride=stride, sample_rate=sample_rate)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, actions_dict
