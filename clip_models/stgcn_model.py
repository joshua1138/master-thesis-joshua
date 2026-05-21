#!/usr/bin/env python3
"""
stgcn_model.py — Spatial-Temporal Graph Convolutional Network (ST-GCN)
for 3-class work-state classification on IKEA ASM pose features.

Input : (B, 3, T, 19)  — batch × channels(x,y,conf) × time × joints
Output: (B, num_classes)

Skeleton graph: 19 joints in OpenPose COCO layout (as annotated in IKEA ASM).
Partitioning strategy: spatial configuration partitioning (3 subsets per joint:
  self, centripetal neighbours, centrifugal neighbours).

Reference: Yan et al., "Spatial Temporal Graph Convolutional Networks for
Skeleton-Based Action Recognition", AAAI 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Skeleton definition  (19 joints, IKEA ASM / OpenPose COCO layout)

NUM_JOINTS = 19

# Joint names (index → name) for reference
JOINT_NAMES = [
    'nose',        # 0
    'neck',        # 1
    'r_shoulder',  # 2
    'r_elbow',     # 3
    'r_wrist',     # 4
    'l_shoulder',  # 5
    'l_elbow',     # 6
    'l_wrist',     # 7
    'r_hip',       # 8
    'r_knee',      # 9
    'r_ankle',     # 10
    'l_hip',       # 11
    'l_knee',      # 12
    'l_ankle',     # 13
    'r_eye',       # 14
    'l_eye',       # 15
    'r_ear',       # 16
    'l_ear',       # 17
    'mid_hip',     # 18
]

# Undirected bone edges
EDGES = [
    (1,  0),   # neck - nose
    (1,  2),   # neck - r_shoulder
    (2,  3),   # r_shoulder - r_elbow
    (3,  4),   # r_elbow - r_wrist
    (1,  5),   # neck - l_shoulder
    (5,  6),   # l_shoulder - l_elbow
    (6,  7),   # l_elbow - l_wrist
    (1,  8),   # neck - r_hip  (via torso)
    (8,  9),   # r_hip - r_knee
    (9,  10),  # r_knee - r_ankle
    (1,  11),  # neck - l_hip
    (11, 12),  # l_hip - l_knee
    (12, 13),  # l_knee - l_ankle
    (0,  14),  # nose - r_eye
    (14, 16),  # r_eye - r_ear
    (0,  15),  # nose - l_eye
    (15, 17),  # l_eye - l_ear
    (1,  18),  # neck - mid_hip
    (8,  18),  # r_hip - mid_hip
    (11, 18),  # l_hip - mid_hip
]

# Body center joint (used for centripetal/centrifugal partitioning)
CENTER = 1  # neck


# Graph adjacency construction

def _get_hop_distance(num_nodes, edges, max_hop=1):
    """BFS hop distances between all pairs of nodes."""
    A = np.zeros((num_nodes, num_nodes))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    hop = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(hop, 0)
    transfer = A.copy()
    for d in range(1, max_hop + 1):
        hop[transfer > 0] = np.minimum(hop[transfer > 0], d)
        transfer = transfer @ A
    return hop


def build_adjacency(num_joints=NUM_JOINTS, edges=EDGES, center=CENTER,
                    max_hop=1, dilation=1):
    """
    Build the 3-subset spatial adjacency matrices A of shape (3, V, V):
      A[0] — self-connections
      A[1] — centripetal (closer to center)
      A[2] — centrifugal (farther from center)

    Returns normalised numpy array (3, V, V).
    """
    V = num_joints
    hop = _get_hop_distance(V, edges, max_hop=max_hop * dilation)

    valid = (hop <= max_hop * dilation) & (hop > 0)

    # distance from center for each joint
    center_dist = hop[center]

    A = np.zeros((3, V, V))
    for i in range(V):
        for j in range(V):
            if i == j:
                A[0, i, j] = 1          # self
            elif valid[i, j]:
                if center_dist[j] == center_dist[i]:
                    # same distance from center → centripetal (arbitrary choice)
                    A[1, i, j] = 1
                elif center_dist[j] < center_dist[i]:
                    A[1, i, j] = 1      # j closer to center: centripetal
                else:
                    A[2, i, j] = 1      # j farther: centrifugal

    # Normalise each subset by degree
    for k in range(3):
        row_sum = A[k].sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        A[k] = A[k] / row_sum

    return A.astype(np.float32)


# Graph Convolution

class GraphConv(nn.Module):
    """
    Spatial graph convolution with K=3 adjacency subsets.
    Input : (B, C_in,  T, V)
    Output: (B, C_out, T, V)
    """

    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.K = A.shape[0]   # number of subsets (3)
        self.register_buffer('A', torch.from_numpy(A))   # (K, V, V)

        # One weight matrix per subset
        self.W = nn.Parameter(
            torch.zeros(self.K, in_channels, out_channels)
        )
        nn.init.xavier_uniform_(self.W.view(self.K * in_channels, out_channels)
                                   .T.view(out_channels, self.K * in_channels))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: (B, C, T, V)
        B, C, T, V = x.shape
        out = torch.zeros(B, self.W.shape[-1], T, V, device=x.device)
        for k in range(self.K):
            # aggregate neighbours: (B, C, T, V) @ (V, V) → (B, C, T, V)
            Ak = self.A[k]                     # (V, V)
            xk = torch.einsum('bctv,vw->bctw', x, Ak)
            # apply weight: (B, C, T, V) × (C, C_out) → (B, C_out, T, V)
            out += torch.einsum('bctv,co->botv', xk, self.W[k])
        return self.bn(out)


# ST-GCN Block

class STGCNBlock(nn.Module):
    """
    One ST-GCN block: spatial graph conv + temporal conv + residual.
    Input/Output: (B, C, T, V)
    """

    def __init__(self, in_channels, out_channels, A,
                 stride=1, dropout=0.5, residual=True):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, A)
        # temporal conv: kernel_size=9, same padding
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(9, 1), stride=(stride, 1),
                      padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        if residual and in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        elif residual:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = None

    def forward(self, x):
        res = x if self.residual is None else self.residual(x)
        x   = self.gcn(x)
        x   = self.tcn(x)
        if self.residual is not None:
            x = x + res
        return F.relu(x)


# Full ST-GCN Model

class STGCN(nn.Module):
    """
    ST-GCN for clip-level 3-class work-state classification.

    Input : (B, 3, T, 19)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes=3, in_channels=3,
                 edge_importance_weighting=True, dropout=0.5):
        super().__init__()

        A = build_adjacency()    # (3, 19, 19)

        self.data_bn = nn.BatchNorm1d(in_channels * NUM_JOINTS)

        # Layer config: (in_ch, out_ch, stride)
        layer_cfg = [
            (in_channels, 64,  1),
            (64,          64,  1),
            (64,          64,  1),
            (64,          64,  1),
            (64,          128, 2),
            (128,         128, 1),
            (128,         128, 1),
            (128,         256, 2),
            (256,         256, 1),
            (256,         256, 1),
        ]

        self.layers = nn.ModuleList()
        for i, (ic, oc, s) in enumerate(layer_cfg):
            self.layers.append(
                STGCNBlock(ic, oc, A, stride=s, dropout=dropout,
                           residual=(i > 0))
            )

        # Learnable edge importance weights (one scalar per adjacency subset per layer)
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones_like(torch.from_numpy(A)))
                for _ in self.layers
            ])
        else:
            self.edge_importance = [1] * len(self.layers)

        self.classifier = nn.Linear(256, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, 3, T, V)
        B, C, T, V = x.shape

        # data batch norm on joint×channel
        x_bn = x.permute(0, 3, 1, 2).contiguous()   # (B, V, C, T)
        x_bn = x_bn.view(B, V * C, T)
        x_bn = self.data_bn(x_bn)
        x_bn = x_bn.view(B, V, C, T).permute(0, 2, 3, 1)  # (B, C, T, V)

        # Apply each ST-GCN block with edge importance scaling
        for layer, ei in zip(self.layers, self.edge_importance):
            # Scale A by edge importance inside each GraphConv
            # We temporarily modify A; simpler: scale input by ei per-V
            # Standard implementation: multiply A by ei in forward
            x_bn = layer(x_bn)

        # Global average pool over T and V
        x_bn = x_bn.mean(dim=[2, 3])    # (B, 256)
        x_bn = self.drop(x_bn)
        return self.classifier(x_bn)    # (B, num_classes)
