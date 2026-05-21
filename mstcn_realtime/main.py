#!/usr/bin/env python3
"""
Main script for causal MS-TCN on 3-class work-state recognition.

"""

import torch
# Trainer is imported conditionally below based on --model_type
from batch_gen import BatchGenerator
import os
import argparse
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train', choices=['train', 'predict'])
parser.add_argument('--model_type', default='causal', choices=['causal', 'acausal'],
                    help='causal=real-time MS-TCN (left-padded); acausal=standard MS-TCN (symmetric padding)')
parser.add_argument('--data_dir', default='./data',
                    help='Path to 3-class data (features/, groundTruth/, splits/, mapping.txt)')
parser.add_argument('--model_dir', default=None,
                    help='Where to save/load models (default: ./models)')
parser.add_argument('--results_dir', default=None,
                    help='Where to save predictions (default: ./results)')
parser.add_argument('--num_stages', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=10)
parser.add_argument('--num_f_maps', type=int, default=64)
parser.add_argument('--features_dim', type=int, default=57,
                    help='Input feature dimension (57 for pose with conf, 38 without)')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--sample_rate', type=int, default=1)

args = parser.parse_args()

# Paths
vid_list_file = os.path.join(args.data_dir, "splits", "train.split.bundle")
vid_list_file_tst = os.path.join(args.data_dir, "splits", "test.split.bundle")
features_path = os.path.join(args.data_dir, "features/")
gt_path = os.path.join(args.data_dir, "groundTruth/")
mapping_file = os.path.join(args.data_dir, "mapping.txt")

model_dir = args.model_dir if args.model_dir else os.path.join(".", "models")
results_dir = args.results_dir if args.results_dir else os.path.join(".", "results")

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Parse mapping file: "0 idle" -> {"idle": 0, ...}
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    if not a.strip():
        continue
    idx, name = a.split(' ', 1)
    actions_dict[name] = int(idx)

num_classes = len(actions_dict)

if args.model_type == 'causal':
    from model import Trainer
    model_label = 'CAUSAL (left-padded dilated convolutions)'
else:
    from model_acausal import Trainer
    model_label = 'ACAUSAL (symmetric-padded dilated convolutions)'

print(f"=== MS-TCN ({args.model_type}) — Work-State Recognition ===")
print(f"  Data dir:      {args.data_dir}")
print(f"  Model dir:     {model_dir}")
print(f"  Results dir:   {results_dir}")
print(f"  Features dim:  {args.features_dim}")
print(f"  Num classes:   {num_classes}  ({', '.join(actions_dict.keys())})")
print(f"  Num stages:    {args.num_stages}")
print(f"  Num layers:    {args.num_layers}")
print(f"  Num f_maps:    {args.num_f_maps}")
print(f"  Batch size:    {args.batch_size}")
print(f"  Learning rate: {args.lr}")
print(f"  Num epochs:    {args.num_epochs}")
print(f"  Sample rate:   {args.sample_rate}")
print(f"  Device:        {device}")
print(f"  Model type:    {model_label}")
print()

trainer = Trainer(args.num_stages, args.num_layers, args.num_f_maps, args.features_dim, num_classes)

if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, args.sample_rate)
    batch_gen.read_data(vid_list_file)
    val_batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, args.sample_rate)
    val_batch_gen.read_data(vid_list_file_tst)
    print(f"Training on {len(batch_gen.list_of_examples)} videos, "
          f"validating on {len(val_batch_gen.list_of_examples)}...")
    log_path = os.path.join(model_dir, 'train_log.csv')
    trainer.train(model_dir, batch_gen, num_epochs=args.num_epochs, batch_size=args.batch_size,
                  learning_rate=args.lr, device=device,
                  val_batch_gen=val_batch_gen, log_path=log_path)
    print(f"Training log: {log_path}")

if args.action == "predict":
    trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst,
                    args.num_epochs, actions_dict, device, args.sample_rate)
