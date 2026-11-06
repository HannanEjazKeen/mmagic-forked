# Copyright (c) OpenMMLab. All rights reserved.
"""Shared helpers for building sonar pix2pix datasets.

Both ``make_paired_sonar.py`` (flat images/ + masks/ folders) and
``make_paired_uatd.py`` (pre-split folders with Pascal-VOC XML boxes) reduce
their input to a list of stems plus a ``loader(stem) -> (sonar, mask)``
callable, then hand it here. :func:`build_dataset` writes two things:

1. an *unpaired* layout, reusable by segmentation / detection models::

       <out>/train/images/<stem>.png   <out>/train/masks/<stem>.png
       <out>/val/images/...            <out>/val/masks/...
       <out>/test/images/...           <out>/test/masks/...

2. a *paired* layout for pix2pix, each file ``[sonar | mask]`` (A|B)::

       <out>/paired/train/<stem>.png
       <out>/paired/val/<stem>.png
       <out>/paired/test/<stem>.png

Point the pix2pix config's ``dataroot`` at ``<out>/paired``.
"""
import os
import os.path as osp
import random
from glob import glob

import mmcv
import numpy as np

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def index_by_stem(folder):
    """Map filename-stem -> full path for every image in a folder."""
    mapping = {}
    for path in sorted(glob(osp.join(folder, '*'))):
        if path.lower().endswith(IMG_EXTS):
            stem = osp.splitext(osp.basename(path))[0]
            mapping[stem] = path
    return mapping


def to_3ch(img):
    """Ensure a loaded image is 3-channel (replicate gray, drop alpha)."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 1:
        img = np.concatenate([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def split_stems(stems, val_ratio, test_ratio, seed):
    """Shuffle ``stems`` and slice into test / val / train by ratio.

    The first ``test_ratio`` fraction becomes test, the next ``val_ratio``
    fraction becomes val, and the remainder becomes train.
    """
    stems = list(stems)
    random.seed(seed)
    random.shuffle(stems)
    n_test = int(round(len(stems) * test_ratio))
    n_val = int(round(len(stems) * val_ratio))
    return {
        'test': stems[:n_test],
        'val': stems[n_test:n_test + n_val],
        'train': stems[n_test + n_val:],
    }


def build_dataset(out_dir, split, loader, size):
    """Write both the unpaired and paired layouts.

    Args:
        out_dir (str): dataroot to create the splits inside.
        split (dict): phase -> list of stems (e.g. from :func:`split_stems`).
        loader (callable): ``stem -> (sonar_bgr, mask_bgr)``; each a 3-channel
            uint8 array of any size (they are resized here).
        size (int): every half is resized to ``size x size``.
    """
    hw = (size, size)
    for phase, phase_stems in split.items():
        img_dir = osp.join(out_dir, phase, 'images')
        mask_dir = osp.join(out_dir, phase, 'masks')
        pair_dir = osp.join(out_dir, 'paired', phase)
        for d in (img_dir, mask_dir, pair_dir):
            os.makedirs(d, exist_ok=True)

        for stem in mmcv.track_iter_progress(phase_stems):
            sonar_img, mask_img = loader(stem)
            # sonar: smooth interpolation; mask: nearest to stay binary/crisp.
            sonar_img = mmcv.imresize(sonar_img, hw)
            mask_img = mmcv.imresize(mask_img, hw, interpolation='nearest')

            mmcv.imwrite(sonar_img, osp.join(img_dir, f'{stem}.png'))
            mmcv.imwrite(mask_img, osp.join(mask_dir, f'{stem}.png'))
            # A|B = sonar | mask  (domain_a | domain_b)
            pair = np.concatenate([sonar_img, mask_img], axis=1)
            mmcv.imwrite(pair, osp.join(pair_dir, f'{stem}.png'))


def report(split, out_dir):
    """Print split sizes and the fake_nums hint for the pix2pix config."""
    print(f'\nDone. train={len(split["train"])}  val={len(split["val"])}  '
          f'test={len(split["test"])}')
    print(f'Unpaired splits: {out_dir}/{{train,val,test}}/{{images,masks}}')
    print(f'Paired (pix2pix dataroot): {out_dir}/paired')
    print(f'Set `fake_nums = {len(split["val"])}` in the pix2pix config '
          f'(val_evaluator runs on val/; keep test/ the same size, or set a '
          f'separate fake_nums for the test_evaluator).')
