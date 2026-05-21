#!/usr/bin/env python3
"""
model_acausal.py — Standard (acausal) MS-TCN.

Drop-in replacement for model.py: same Trainer interface, same forward
signature.  main_acausal.py imports Trainer from this file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np


class DilatedResidualLayer(nn.Module):
    """Dilated residual layer with SYMMETRIC ("same") padding."""

    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        # kernel_size=3, symmetric padding=dilation keeps T fixed
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                                       padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)
             for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList(
            [copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes))
             for s in range(num_stages - 1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device,
              val_batch_gen=None, log_path=None):
        """Train with optional validation tracking (same signature as model.py)."""
        import time, csv as _csv
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        if val_batch_gen is not None and log_path is not None:
            with open(log_path, 'w', newline='') as f:
                _csv.writer(f).writerow(
                    ['epoch', 'train_loss', 'train_acc',
                     'val_loss', 'val_acc', 'time_s'])

        for epoch in range(num_epochs):
            t0 = time.time()
            self.model.train()
            epoch_loss = 0
            correct = 0
            total = 0

            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                mask = mask.to(device)

                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                                   batch_target.view(-1))
                    loss += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1),
                                 F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                        min=0, max=16) * mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            train_loss = epoch_loss / len(batch_gen.list_of_examples)
            train_acc = float(correct) / total
            elapsed = time.time() - t0

            val_loss, val_acc = float('nan'), float('nan')
            if val_batch_gen is not None:
                self.model.eval()
                v_loss_sum, v_correct, v_total = 0.0, 0, 0
                with torch.no_grad():
                    while val_batch_gen.has_next():
                        vin, vtgt, vmask = val_batch_gen.next_batch(batch_size)
                        vin, vtgt, vmask = vin.to(device), vtgt.to(device), vmask.to(device)
                        vpred = self.model(vin, vmask)
                        v_l = 0
                        for p in vpred:
                            v_l += self.ce(
                                p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                                vtgt.view(-1))
                            v_l += 0.15 * torch.mean(torch.clamp(
                                self.mse(F.log_softmax(p[:, :, 1:], dim=1),
                                         F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                                min=0, max=16) * vmask[:, :, 1:])
                        v_loss_sum += v_l.item()
                        _, vp = torch.max(vpred[-1].data, 1)
                        v_correct += ((vp == vtgt).float() * vmask[:, 0, :].squeeze(1)).sum().item()
                        v_total += torch.sum(vmask[:, 0, :]).item()
                val_batch_gen.reset()
                val_loss = v_loss_sum / max(1, len(val_batch_gen.list_of_examples))
                val_acc = v_correct / max(1, v_total)

            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")

            if val_batch_gen is None:
                print("[epoch %d]: epoch loss = %f,   acc = %f" % (
                    epoch + 1, train_loss, train_acc))
            else:
                print("[epoch %d]: tr_loss=%.4f tr_acc=%.4f  "
                      "val_loss=%.4f val_acc=%.4f  %.1fs" % (
                          epoch + 1, train_loss, train_acc,
                          val_loss, val_acc, elapsed))
                if log_path is not None:
                    with open(log_path, 'a', newline='') as f:
                        _csv.writer(f).writerow([
                            epoch + 1, round(train_loss, 5), round(train_acc, 5),
                            round(val_loss, 5), round(val_acc, 5),
                            round(elapsed, 1)])

    def predict(self, model_dir, results_dir, features_path, vid_list_file,
                epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(
                torch.load(model_dir + "/epoch-" + str(epoch) + ".model",
                           map_location=device))

            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            list_of_vids = [v for v in list_of_vids if v.strip()]
            file_ptr.close()

            idx_to_action = {v: k for k, v in actions_dict.items()}

            import os
            probs_dir = os.path.join(results_dir, "probs")
            os.makedirs(probs_dir, exist_ok=True)

            for vid in list_of_vids:
                print(vid)
                features = np.load(features_path + vid + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)

                predictions = self.model(input_x, torch.ones(input_x.size(), device=device))

                probs = F.softmax(predictions[-1], dim=1).squeeze(0).cpu().numpy()

                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()

                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((
                        recognition,
                        [idx_to_action[predicted[i].item()]] * sample_rate))

                f_name = vid.split('/')[-1]

                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()

                if sample_rate > 1:
                    probs = np.repeat(probs, sample_rate, axis=1)
                np.save(os.path.join(probs_dir, f_name + ".npy"), probs)
