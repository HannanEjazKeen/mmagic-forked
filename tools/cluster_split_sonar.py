# Copyright (c) OpenMMLab. All rights reserved.
"""Leakage-free, group-aware split for the sonar pix2pix dataset.

The marine-debris images are sampled from video, so many frames are near
duplicates. A random train/val/test split scatters those near-duplicates
across splits, letting pix2pix reproduce content it effectively memorised
(data leakage). This script removes that leakage:

1. embed every sonar image with a pretrained CNN (default ResNet-18);
2. connect any pair of images with cosine similarity >= --sim-threshold and
   take the CONNECTED COMPONENTS as groups (a drifting chain A~B~C becomes one
   group even if A and C are not directly similar);
3. assign whole groups to train / val / test, so no group ever straddles a
   split. By construction, no cross-split pair exceeds the threshold.

Run with --dry-run first to inspect group sizes and the worst remaining
train<->test similarity, tune --sim-threshold, then run for real to write the
paired dataset (same layout as tools/make_paired_sonar.py, via
tools/sonar_dataset_utils.py).

Example
-------
# 1) inspect grouping without writing anything
python tools/cluster_split_sonar.py \
    --sonar-dir /path/sonar --mask-dir /path/masks \
    --sim-threshold 0.9 --dry-run

# 2) write the leakage-free paired dataset
python tools/cluster_split_sonar.py \
    --sonar-dir /path/sonar --mask-dir /path/masks \
    --out-dir data/pix2pix/sonar --sim-threshold 0.9 \
    --val-ratio 0.1 --test-ratio 0.1 --seed 0
"""
import argparse
import random

import mmcv
import numpy as np
import torch
import torchvision.transforms.functional as TF
from mmengine.utils import track_iter_progress
from torchvision import models

from sonar_dataset_utils import build_dataset, index_by_stem, report, to_3ch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Leakage-free grouped split for sonar pix2pix.')
    parser.add_argument('--sonar-dir', required=True, help='sonar images')
    parser.add_argument('--mask-dir', required=True, help='binary masks')
    parser.add_argument('--out-dir', default='data/pix2pix/sonar')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--sim-threshold',
        type=float,
        default=0.9,
        help='cosine similarity >= this merges two images into one group')
    parser.add_argument('--embed-arch', default='resnet18')
    parser.add_argument('--embed-batch', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='only report grouping/leakage stats; write nothing')
    return parser.parse_args()


@torch.no_grad()
def embed_images(stems, sonar_paths, arch, batch, device):
    """L2-normalised penultimate-layer features for each sonar image."""
    net = getattr(models, arch)(weights='DEFAULT')
    net.fc = torch.nn.Identity()  # drop classifier -> feature vector
    net.eval().to(device)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    feats = []
    for i in track_iter_progress(list(range(0, len(stems), batch))):
        chunk = stems[i:i + batch]
        tensors = []
        for stem in chunk:
            img = to_3ch(mmcv.imread(sonar_paths[stem], flag='color'))
            img = mmcv.imresize(img, (224, 224))[:, :, ::-1]  # BGR->RGB
            t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
            t = TF.normalize(t.float() / 255.0, mean, std)
            tensors.append(t)
        out = net(torch.stack(tensors).to(device))
        feats.append(torch.nn.functional.normalize(out, dim=1).cpu())
    return torch.cat(feats)  # (N, D), unit norm


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def connected_components(feats, threshold, device):
    """Group indices whose cosine similarity >= threshold (transitively)."""
    n = feats.shape[0]
    uf = UnionFind(n)
    feats = feats.to(device)
    # blockwise NxN cosine sim (feats are unit norm, so dot == cosine)
    block = 1024
    for i in range(0, n, block):
        sim = feats[i:i + block] @ feats.T  # (b, N)
        rows, cols = torch.where(sim >= threshold)
        for r, c in zip(rows.tolist(), cols.tolist()):
            gi = i + r
            if gi < c:  # upper triangle only, skip self
                uf.union(gi, c)
    comp = {}
    for idx in range(n):
        comp.setdefault(uf.find(idx), []).append(idx)
    return list(comp.values())


def grouped_split(groups, n, val_ratio, test_ratio, seed):
    """Assign whole groups to test / val / train by target ratios."""
    rng = random.Random(seed)
    order = groups[:]
    rng.shuffle(order)
    n_test, n_val = round(n * test_ratio), round(n * val_ratio)
    split = {'test': [], 'val': [], 'train': []}
    for grp in order:
        if len(split['test']) < n_test:
            split['test'] += grp
        elif len(split['val']) < n_val:
            split['val'] += grp
        else:
            split['train'] += grp
    return split


def nn_similarity(feats, device):
    """Per-image highest similarity to any OTHER image (nearest neighbour)."""
    n = feats.shape[0]
    feats = feats.to(device)
    nn = torch.empty(n)
    for i in range(0, n, 1024):
        sim = feats[i:i + 1024] @ feats.T
        # mask self-similarity on the diagonal
        for r in range(sim.shape[0]):
            sim[r, i + r] = -1.0
        nn[i:i + 1024] = sim.max(dim=1).values.cpu()
    return nn


def sweep_thresholds(feats, thresholds, device):
    """Report group count / largest group for a range of thresholds."""
    n = feats.shape[0]
    print('\n  thr   groups   max_group   max_group%')
    for thr in thresholds:
        comps = connected_components(feats, thr, device)
        sizes = sorted((len(c) for c in comps), reverse=True)
        print(f'  {thr:.2f}  {len(comps):>6}  {sizes[0]:>9}   '
              f'{100 * sizes[0] / n:5.1f}%')


def main():
    args = parse_args()

    sonar = index_by_stem(args.sonar_dir)
    masks = index_by_stem(args.mask_dir)
    stems = sorted(set(sonar) & set(masks))
    if not stems:
        raise RuntimeError('No matching stems between --sonar-dir/--mask-dir.')
    print(f'{len(stems)} matched image/mask pairs.')

    feats = embed_images(stems, sonar, args.embed_arch, args.embed_batch,
                         args.device)
    # Remove the large component shared by all sonar images (ImageNet features
    # otherwise rate every pair >0.9). Centering makes cosine behave like
    # Pearson correlation, so a threshold actually separates near-duplicates.
    feats = feats - feats.mean(dim=0, keepdim=True)
    feats = torch.nn.functional.normalize(feats, dim=1)

    # Distribution of each image's nearest-neighbour similarity: near-dups sit
    # near 1.0, isolated frames much lower. Use it to read off a threshold.
    nn = nn_similarity(feats, args.device)
    pct = [50, 75, 90, 95, 99]
    qs = np.percentile(nn.numpy(), pct)
    print('\n--- nearest-neighbour similarity (after centering) ---')
    print('  ' + '  '.join(f'p{p}={q:.3f}' for p, q in zip(pct, qs)))
    print(f'  min={nn.min():.3f}  max={nn.max():.3f}')

    if args.dry_run:
        sweep_thresholds(feats, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95], args.device)
        print(f'\nAt your chosen --sim-threshold={args.sim_threshold}:')

    comps = connected_components(feats, args.sim_threshold, args.device)
    comps.sort(key=len, reverse=True)
    sizes = [len(c) for c in comps]
    print(f'  groups: {len(comps)}  singletons: {sum(s == 1 for s in sizes)}  '
          f'max group: {sizes[0]} ({100 * sizes[0] / len(stems):.1f}%)  '
          f'mean: {np.mean(sizes):.1f}')
    if sizes[0] > len(stems) // 2:
        print('  [warn] one group holds >50% of the data -- threshold too low '
              '(single-linkage chaining). Raise --sim-threshold.')

    stem_groups = [[stems[i] for i in c] for c in comps]
    split = grouped_split(stem_groups, len(stems), args.val_ratio,
                         args.test_ratio, args.seed)
    print(f'  split -> train={len(split["train"])} val={len(split["val"])} '
          f'test={len(split["test"])}')

    # leakage check: worst train<->test similarity (should be < threshold)
    if split['train'] and split['test']:
        idx_of = {s: i for i, s in enumerate(stems)}
        tr = feats[[idx_of[s] for s in split['train']]].to(args.device)
        te = feats[[idx_of[s] for s in split['test']]].to(args.device)
        worst = max((te[i:i + 1024] @ tr.T).max().item()
                    for i in range(0, te.shape[0], 1024))
        print(f'  worst train<->test similarity: {worst:.4f} '
              f'(must be < {args.sim_threshold})')
    else:
        print('  [warn] a split is empty -- threshold too low; nothing usable.')

    if args.dry_run:
        print('\n[dry-run] nothing written. Pick --sim-threshold from the '
              'sweep/percentiles above, then re-run without --dry-run.')
        return
    if not (split['train'] and split['val'] and split['test']):
        raise RuntimeError('Empty split at this threshold; raise '
                           '--sim-threshold and retry.')

    def loader(stem):
        s = to_3ch(mmcv.imread(sonar[stem], flag='color'))
        m = to_3ch(mmcv.imread(masks[stem], flag='color'))
        return s, m

    build_dataset(args.out_dir, split, loader, args.size)
    report(split, args.out_dir)


if __name__ == '__main__':
    main()
