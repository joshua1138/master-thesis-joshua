#!/usr/bin/env python3
"""
setup_data.py — Cross-platform replacement for setup_data.sh.

Prepares the `data/` directory for a 3-class experiment by:
  1. Linking features/ and splits/ from the original 33-class dataset
  2. Running relabel_ground_truth.py to produce 3-class groundTruth/

Usage (Windows):
    python setup_data.py "C:\\Users\\PC\\Desktop\\Master Thesis\\ikea\\mstcn_data"

Run from the project folder (e.g. ikea/mstcn_realtime/ or ikea/clip_models/).
The script creates ./data/ relative to its own location.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def link_dir(src: Path, dst: Path):
    """Create a directory link from dst -> src. Cross-platform."""
    if dst.exists() or dst.is_symlink():
        print(f"  {dst.name}/ already exists, skipping")
        return

    src = src.resolve()

    if os.name == 'nt':
        # Windows: directory junction (no admin required)
        # mklink /J <link> <target>
        result = subprocess.run(
            ['cmd', '/c', 'mklink', '/J', str(dst), str(src)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ⚠ Junction failed ({result.stderr.strip()}). "
                  f"Falling back to copy — this will use disk space.")
            shutil.copytree(src, dst)
            print(f"  ✓ Copied {src} → {dst}")
        else:
            print(f"  ✓ Junction {dst} → {src}")
    else:
        # Linux / macOS: symlink
        os.symlink(src, dst)
        print(f"  ✓ Symlinked {dst} → {src}")


def main():
    parser = argparse.ArgumentParser(
        description='Set up 3-class data directory (cross-platform).'
    )
    parser.add_argument('orig_data',
                        help='Path to original 33-class data dir (with features/, '
                             'groundTruth/, splits/, mapping.txt)')
    parser.add_argument('--data_dir', default='./data',
                        help='Output data dir (default: ./data)')
    args = parser.parse_args()

    orig = Path(args.orig_data).resolve()
    data_dir = Path(args.data_dir).resolve()
    script_dir = Path(__file__).parent.resolve()

    print(f"Original data: {orig}")
    print(f"New data dir:  {data_dir}\n")

    # Validate source layout
    for sub in ['features', 'groundTruth', 'splits']:
        if not (orig / sub).is_dir():
            print(f"ERROR: {orig / sub} not found.")
            sys.exit(1)
    if not (orig / 'mapping.txt').is_file():
        print(f"ERROR: {orig / 'mapping.txt'} not found.")
        sys.exit(1)

    data_dir.mkdir(parents=True, exist_ok=True)

    # Link features and splits (these don't change between 33-class and 3-class)
    link_dir(orig / 'features', data_dir / 'features')
    link_dir(orig / 'splits',   data_dir / 'splits')

    # Relabel groundTruth: 33 → 3 classes
    relabel_script = script_dir / 'relabel_ground_truth.py'
    if not relabel_script.is_file():
        print(f"\nERROR: {relabel_script} not found in script directory.")
        print("       Make sure relabel_ground_truth.py is alongside setup_data.py.")
        sys.exit(1)

    print("\nRelabeling ground truth (33 → 3 classes)...")
    cmd = [
        sys.executable, str(relabel_script),
        '--src_gt_dir',  str(orig / 'groundTruth') + os.sep,
        '--dst_gt_dir',  str(data_dir / 'groundTruth') + os.sep,
        '--dst_mapping', str(data_dir / 'mapping.txt'),
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: relabel_ground_truth.py failed.")
        sys.exit(result.returncode)

    print("\n=== Setup complete ===")
    print(f"You can now train with:")
    print(f"  python main.py --action train --data_dir {data_dir}")


if __name__ == '__main__':
    main()
