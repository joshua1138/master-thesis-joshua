#!/usr/bin/env python3
"""
official_hcn.py — Ben-Shabat et al.'s HCN setup for IKEA ASM.


Input shape : (N, C, T, V, M)   N=batch, C=2 (x,y), T=window, V=joints, M=1
Output shape: (N, num_class)    — single class score per clip

The window must be divisible by 16 (because fc7 expects (window/16)^2 features).
For window=32 → fc7 input is 256*2*2; for window=64 → 256*4*4.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _weights_init(m):
    """Match the initialisation used in the official HCN repo (Glorot uniform)."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        ws = list(m.weight.data.size())
        fan_in = ws[1] * ws[2] * ws[3] if len(ws) == 4 else ws[1] * ws[2]
        fan_out = ws[0] * ws[2] * ws[3] if len(ws) == 4 else ws[0] * ws[2]
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-bound, bound)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        ws = list(m.weight.data.size())
        bound = math.sqrt(6.0 / (ws[0] + ws[1]))
        m.weight.data.uniform_(-bound, bound)
        if m.bias is not None:
            m.bias.data.fill_(0)


def _init_all(module):
    for child in module.children():
        if list(child.children()):
            _init_all(child)
        else:
            _weights_init(child)


class HCN(nn.Module):

    def __init__(self, in_channel=2, num_joint=18, num_person=1,
                 out_channel=64, window_size=32, num_class=33):
        super().__init__()
        self.num_person = num_person
        self.num_class = num_class
        self.window_size = window_size
        assert window_size % 16 == 0, "window_size must be divisible by 16"

        # === Position stream ===
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv2d(out_channel, window_size,
                               kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_joint, out_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
        )

        # === Motion stream (mirrors position) ===
        self.conv1m = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.conv2m = nn.Conv2d(out_channel, window_size,
                                kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.conv3m = nn.Sequential(
            nn.Conv2d(num_joint, out_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
        )
        self.conv4m = nn.Sequential(
            nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
        )

        # === Fusion ===
        self.conv5 = nn.Sequential(
            nn.Conv2d(out_channel * 2, out_channel * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(out_channel * 2, out_channel * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
        )

        # fc7 input: out_channel*4 channels × (window/16)² spatial
        fc7_in = (out_channel * 4) * (window_size // 16) * (window_size // 16)
        self.fc7 = nn.Sequential(
            nn.Linear(fc7_in, 256 * 2),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )
        self.fc8 = nn.Linear(256 * 2, num_class)

        _init_all(self)

    def forward(self, x, target=None):
        # x: (N, C, T, V, M)
        N, C, T, V, M = x.size()

        # Motion: first-difference along T, upsample back to T
        motion = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        motion = motion.permute(0, 1, 4, 2, 3).contiguous().view(N, C * M, T - 1, V)
        motion = F.interpolate(motion, size=(T, V), mode='bilinear',
                               align_corners=False)
        motion = motion.view(N, C, M, T, V).permute(0, 1, 3, 4, 2)

        logits = []
        for i in range(self.num_person):
            # Position stream
            out = self.conv1(x[:, :, :, :, i])
            out = self.conv2(out)
            out = out.permute(0, 3, 2, 1).contiguous()
            out = self.conv3(out)
            out_p = self.conv4(out)

            # Motion stream
            out = self.conv1m(motion[:, :, :, :, i])
            out = self.conv2m(out)
            out = out.permute(0, 3, 2, 1).contiguous()
            out = self.conv3m(out)
            out_m = self.conv4m(out)

            # Concatenate streams
            out = torch.cat((out_p, out_m), dim=1)
            out = self.conv5(out)
            out = self.conv6(out)
            logits.append(out)

        out = logits[0]  # single-person assumption (M=1)
        out = out.view(out.size(0), -1)
        out = self.fc7(out)
        out = self.fc8(out)
        return out
