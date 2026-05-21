#!/usr/bin/env python3
"""
hcn_model.py — Hierarchical Co-occurrence Network (HCN)
for 3-class work-state classification on IKEA ASM pose features.

Input : (B, 3, T, 19)  — same format as ST-GCN (permuted internally)
Output: (B, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HCNStream(nn.Module):

    def __init__(self, in_channels=3, num_joints=19,
                 stage1_channels=32, stage2_channels=64):
        super().__init__()
        # Stage 1: conv over joints dimension for each time step
        # Treats input as (B*T, C, V) and applies 1D conv along V
        self.stage1 = nn.Sequential(
            nn.Conv1d(in_channels, stage1_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm1d(stage1_channels),
            nn.ReLU(),
            nn.Conv1d(stage1_channels, stage1_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(stage1_channels),
            nn.ReLU(),
        )

        # Stage 2: 2D conv over (joints × time)
        # After stage1 output is (B, stage1_ch, T, V) → treat as image
        self.stage2 = nn.Sequential(
            nn.Conv2d(stage1_channels, stage2_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stage2_channels),
            nn.ReLU(),
            nn.Conv2d(stage2_channels, stage2_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stage2_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),   # halve T and V
        )

    def forward(self, x):
        # x: (B, T, V, C)
        B, T, V, C = x.shape

        # Stage 1: apply per-time-step along joint dimension
        # Reshape to (B*T, C, V)
        x1 = x.reshape(B * T, V, C).permute(0, 2, 1)  # (B*T, C, V)
        x1 = self.stage1(x1)                           # (B*T, ch1, V)
        ch1 = x1.shape[1]

        # Reshape back to (B, T, ch1, V) → permute to (B, ch1, T, V)
        x1 = x1.view(B, T, ch1, V).permute(0, 2, 1, 3)  # (B, ch1, T, V)

        # Stage 2: 2D conv over (T, V)
        x2 = self.stage2(x1)   # (B, ch2, T//2, V//2)

        return x2


class HCN(nn.Module):
    """
    Hierarchical Co-occurrence Network for clip-level classification.

    Input : (B, 3, T, V)  — channels × time × joints  (same as ST-GCN loader)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes=3, in_channels=3, num_joints=19,
                 stage1_channels=32, stage2_channels=64,
                 fc_hidden=256, dropout=0.5):
        super().__init__()

        self.pos_stream = HCNStream(in_channels, num_joints,
                                    stage1_channels, stage2_channels)
        self.vel_stream = HCNStream(in_channels, num_joints,
                                    stage1_channels, stage2_channels)

        # After global avg pooling both streams and concatenating:
        # each stream contributes stage2_channels features
        combined = stage2_channels * 2

        self.classifier = nn.Sequential(
            nn.Linear(combined, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.BatchNorm1d(fc_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden // 2, num_classes),
        )

    def forward(self, x):
        # x: (B, C, T, V) — from loader
        # Permute to (B, T, V, C) for HCN streams
        x = x.permute(0, 2, 3, 1)    # (B, T, V, C)

        # Velocity: frame differences (pad first frame to keep T the same)
        vel = torch.zeros_like(x)
        vel[:, 1:, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]

        pos_feat = self.pos_stream(x)    # (B, ch2, T', V')
        vel_feat = self.vel_stream(vel)  # (B, ch2, T', V')

        # Global average pooling over T and V
        pos_feat = pos_feat.mean(dim=[2, 3])   # (B, ch2)
        vel_feat = vel_feat.mean(dim=[2, 3])   # (B, ch2)

        combined = torch.cat([pos_feat, vel_feat], dim=1)  # (B, ch2*2)

        return self.classifier(combined)   # (B, num_classes)
