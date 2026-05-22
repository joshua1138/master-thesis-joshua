#!/usr/bin/env python3

import torch
from model import Trainer
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
parser.add_argument('--data_dir', default='./data/ikea_asm',
                    help='Path to preprocessed MS-TCN data (features/, groundTruth/, splits/, mapping.txt)')
parser.add_argument('--model_dir', default=None,
                    help='Where to save/load models (default: ./models/ikea_asm)')
parser.add_argument('--results_dir', default=None,
                    help='Where to save predictions (default: ./results/ikea_asm)')
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

model_dir = args.model_dir if args.model_dir else os.path.join("models", "ikea_asm")
results_dir = args.results_dir if args.results_dir else os.path.join("results", "ikea_asm")

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Parse mapping file: "0 NA" -> {"NA": 0}
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    idx, name = a.split(' ', 1)
    actions_dict[name] = int(idx)

num_classes = len(actions_dict)

print(f"Dataset: IKEA ASM")
print(f"  Data dir:      {args.data_dir}")
print(f"  Model dir:     {model_dir}")
print(f"  Results dir:   {results_dir}")
print(f"  Features dim:  {args.features_dim}")
print(f"  Num classes:   {num_classes}")
print(f"  Num stages:    {args.num_stages}")
print(f"  Num layers:    {args.num_layers}")
print(f"  Num f_maps:    {args.num_f_maps}")
print(f"  Batch size:    {args.batch_size}")
print(f"  Learning rate: {args.lr}")
print(f"  Num epochs:    {args.num_epochs}")
print(f"  Sample rate:   {args.sample_rate}")
print(f"  Device:        {device}")
print()

trainer = Trainer(args.num_stages, args.num_layers, args.num_f_maps, args.features_dim, num_classes)

if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, args.sample_rate)
    batch_gen.read_data(vid_list_file)
    print(f"Training on {len(batch_gen.list_of_examples)} videos...")
    trainer.train(model_dir, batch_gen, num_epochs=args.num_epochs, batch_size=args.batch_size,
                  learning_rate=args.lr, device=device)

if args.action == "predict":
    trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst,
                    args.num_epochs, actions_dict, device, args.sample_rate)
