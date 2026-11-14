"""Count the boxes in each split of the detector dataset and look for images shared between splits.

    python tools/check_dataset.py path/to/Sinkhole_Vision_Fusion-1

Two images count as near-duplicates when their 16x16 average hashes differ in
at most 5% of the bits. This is a coarse check: it finds copies and resized
copies, not two different photos of the same place.
"""
import collections
import glob
import os
import sys

import cv2
import numpy as np

CLASS_NAMES = {0: 'Concentric_Crack', 1: 'Pothole'}
MAX_DIFFERENT_BITS = 13      # 5% of 256


def average_hash(path):
    gray = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (16, 16), interpolation=cv2.INTER_AREA)
    return (gray > gray.mean()).flatten()


def box_counts(label_dir):
    counts = collections.Counter()
    for path in glob.glob(os.path.join(label_dir, '*.txt')):
        for line in open(path).read().splitlines():
            if line.strip():
                counts[CLASS_NAMES.get(int(line.split()[0]), 'other')] += 1
    return dict(counts)


def near_duplicates(images, reference_images):
    reference = [average_hash(p) for p in reference_images]
    found = 0
    for path in images:
        h = average_hash(path)
        if any(int((h != r).sum()) <= MAX_DIFFERENT_BITS for r in reference):
            found += 1
    return found


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'Sinkhole_Vision_Fusion-1'
    images = {s: sorted(glob.glob(os.path.join(root, s, 'images', '*.jpg'))) for s in ('train', 'valid', 'test')}
    for split, paths in images.items():
        print(f'{split:5s}: {len(paths):3d} images, boxes {box_counts(os.path.join(root, split, "labels"))}')
    for split in ('valid', 'test'):
        n = near_duplicates(images[split], images['train'])
        print(f'{split}: {n} of {len(images[split])} images have a near-duplicate in train')


if __name__ == '__main__':
    main()
