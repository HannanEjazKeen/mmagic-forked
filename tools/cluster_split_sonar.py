# Copyright (c) OpenMMLab. All rights reserved.
"""Leakage-free, group-aware split for the sonar pix2pix dataset (SSIM-based).

The marine-debris images are sampled from video, so many frames are near
duplicates. A random train/val/test split scatters those near-duplicates
across splits, letting pix2pix reproduce content it effectively memorised
(data leakage). This script removes that leakage with a two-stage grouping:

1. CANDIDATES (cheap recall): embed every sonar image with a pretrained CNN
   and, for each image, take its top-K most similar images by feature cosine.
   This avoids computing SSIM for all ~N^2 pairs.
2. DECISION (precise): compute SSIM between each candidate pair on small
   grayscale images; connect pairs with SSIM >= --ssim-threshold and take the
   CONNECTED COMPONENTS as groups. SSIM compares pixel structure, so true
   duplicate frames score near 1 and distinct scenes drop off sharply.
3. assign whole groups to train / val / test, so no group straddles a split.

Run with --dry-run first to see the SSIM distribution over candidate pairs and
a threshold sweep, pick --ssim-threshold, then run for real to write the
paired dataset (same layout as tools/make_paired_sonar.py).

Example
-------
python tools/cluster_split_sonar.py \
    --sonar-dir /path/sonar --mask-dir /path/masks \
    --ssim-threshold 0.6 --dry-run
"""
import argparse
import random

import mmcv
import numpy as np
import torch
import torchvision.transforms.functional as TF
from mmengine.utils import track_iter_progress
from skimage.metrics import structural_similarity as ssim
from torchvision import models

from sonar_dataset_utils import build_dataset, index_by_stem, report, to_3ch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Leakage-free SSIM-based grouped split for sonar pix2pix.')
    parser.add_argument('--sonar-dir', required=True, help='sonar images')
    parser.add_argument('--mask-dir', required=True, help='binary masks')
    parser.add_argument('--out-dir', default='data/pix2pix/sonar')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--ssim-threshold',
        type=float,
        default=0.6,
        help='SSIM >= this merges two images into one group')
    parser.add_argument(
        '--topk',
        type=int,
        default=20,
        help='SSIM is computed only against each image top-K feature neighbours')
    parser.add_argument(
        '--ssim-size',
        type=int,
        default=128,
        help='images are resized to this (grayscale) for SSIM')
    parser.add_argument('--embed-arch', default='resnet18')
    parser.add_argument('--embed-batch', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


@torch.no_grad()
def embed_images(stems, sonar_paths, arch, batch, device):
    """L2-normalised, mean-centred penultimate features per sonar image."""
    net = getattr(models, arch)(weights='DEFAULT')
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    feats = []
    print('Embedding images for candidate neighbours...')
    for i in track_iter_progress(list(range(0, len(stems), batch))):
        tensors = []
        for stem in stems[i:i + batch]:
            img = to_3ch(mmcv.imread(sonar_paths[stem], flag='color'))
            img = mmcv.imresize(img, (224, 224))[:, :, ::-1]
            t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
            tensors.append(TF.normalize(t.float() / 255.0, mean, std))
        feats.append(net(torch.stack(tensors).to(device)).cpu())
    feats = torch.cat(feats)
    feats = feats - feats.mean(dim=0, keepdim=True)
    return torch.nn.functional.normalize(feats, dim=1)


def load_gray(stems, sonar_paths, size):
    """Small grayscale float images for SSIM, keyed by index."""
    print('Loading small grayscale images for SSIM...')
    imgs = []
    for stem in track_iter_progress(stems):
        g = mmcv.imread(sonar_paths[stem], flag='grayscale')
        imgs.append(mmcv.imresize(g, (size, size)).astype(np.float64))
    return imgs


def candidate_pairs(feats, topk, device):
    """Unique (i, j) pairs from each image top-K feature neighbours."""
    n = feats.shape[0]
    feats = feats.to(device)
    pairs = set()
    for i in range(0, n, 1024):
        sim = feats[i:i + 1024] @ feats.T
        idx = sim.topk(min(topk + 1, n), dim=1).indices.cpu().numpy()
        for r, neighbours in enumerate(idx):
            gi = i + r
            for j in neighbours:
                if gi != j:
                    pairs.add((min(gi, int(j)), max(gi, int(j))))
    return sorted(pairs)


def ssim_for_pairs(pairs, gray):
    """SSIM for each candidate pair."""
    print(f'Computing SSIM for {len(pairs)} candidate pairs...')
    scores = np.empty(len(pairs))
    for k, (i, j) in enumerate(track_iter_progress(pairs)):
        scores[k] = ssim(gray[i], gray[j], data_range=255.0)
    return scores


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


def components(n, pairs, scores, threshold):
    """Connected components over pairs whose SSIM >= threshold."""
    uf = UnionFind(n)
    for (i, j), s in zip(pairs, scores):
        if s >= threshold:
            uf.union(i, j)
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


def main():
    args = parse_args()

    sonar = index_by_stem(args.sonar_dir)
    masks = index_by_stem(args.mask_dir)
    stems = sorted(set(sonar) & set(masks))
    if not stems:
        raise RuntimeError('No matching stems between --sonar-dir/--mask-dir.')
    n = len(stems)
    print(f'{n} matched image/mask pairs.')

    feats = embed_images(stems, sonar, args.embed_arch, args.embed_batch,
                         args.device)
    gray = load_gray(stems, sonar, args.ssim_size)
    pairs = candidate_pairs(feats, args.topk, args.device)
    scores = ssim_for_pairs(pairs, gray)

    qs = np.percentile(scores, [50, 75, 90, 95, 99])
    print('\n--- SSIM over candidate pairs ---')
    print('  ' + '  '.join(f'p{p}={q:.3f}'
                           for p, q in zip([50, 75, 90, 95, 99], qs)))
    print(f'  min={scores.min():.3f}  max={scores.max():.3f}')

    if args.dry_run:
        print('\n  thr   groups   max_group   max_group%')
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            comps = components(n, pairs, scores, thr)
            big = max(len(c) for c in comps)
            print(f'  {thr:.2f}  {len(comps):>6}  {big:>9}   '
                  f'{100 * big / n:5.1f}%')
        print(f'\nAt your chosen --ssim-threshold={args.ssim_threshold}:')

    comps = components(n, pairs, scores, args.ssim_threshold)
    comps.sort(key=len, reverse=True)
    sizes = [len(c) for c in comps]
    print(f'  groups: {len(comps)}  singletons: {sum(s == 1 for s in sizes)}  '
          f'max group: {sizes[0]} ({100 * sizes[0] / n:.1f}%)  '
          f'mean: {np.mean(sizes):.1f}')

    stem_groups = [[stems[i] for i in c] for c in comps]
    split = grouped_split(stem_groups, n, args.val_ratio, args.test_ratio,
                         args.seed)
    print(f'  split -> train={len(split["train"])} val={len(split["val"])} '
          f'test={len(split["test"])}')

    # leakage check over cross-split candidate pairs (built to be < threshold)
    split_of = {}
    for ph, ss in split.items():
        for s in ss:
            split_of[s] = ph
    worst = 0.0
    for (i, j), s in zip(pairs, scores):
        if split_of[stems[i]] != split_of[stems[j]]:
            worst = max(worst, s)
    print(f'  worst cross-split SSIM (candidate pairs): {worst:.4f} '
          f'(must be < {args.ssim_threshold})')

    if args.dry_run:
        print('\n[dry-run] nothing written. Pick --ssim-threshold from the '
              'sweep above, then re-run without --dry-run.')
        return
    if not (split['train'] and split['val'] and split['test']):
        raise RuntimeError('Empty split; adjust --ssim-threshold and retry.')

    def loader(stem):
        s = to_3ch(mmcv.imread(sonar[stem], flag='color'))
        m = to_3ch(mmcv.imread(masks[stem], flag='color'))
        return s, m

    build_dataset(args.out_dir, split, loader, args.size)
    report(split, args.out_dir)


if __name__ == '__main__':
    main()
