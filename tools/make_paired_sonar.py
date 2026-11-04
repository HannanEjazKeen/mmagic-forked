# Copyright (c) OpenMMLab. All rights reserved.
"""Build a pix2pix paired dataset from separate sonar images and binary masks.

Each output file is a single image with the sonar image on the LEFT and the
mask on the RIGHT: ``[sonar | mask]`` (A|B, concatenated along the width).
This matches the ``domain_a=sonar`` / ``domain_b=mask`` convention used by
``configs/pix2pix/pix2pix_vanilla-unet-bn_1xb1-80kiters_sonar.py`` (mmagic),
where the generator takes the mask and produces the sonar image.

Pairs are matched by filename stem: ``sonar/0001.png`` <-> ``masks/0001.png``.
The matched pairs are shuffled and split into ``train/``, ``val/`` and
``test/``.

Example
-------
python tools/make_paired_sonar.py \
    --sonar-dir /path/to/sonar_images \
    --mask-dir  /path/to/binary_masks \
    --out-dir   data/pix2pix/sonar \
    --size 256 --val-ratio 0.1 --test-ratio 0.1 --seed 0
"""
import argparse
import os
import os.path as osp
import random
from glob import glob

import mmcv
import numpy as np

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Make a pix2pix [sonar|mask] paired dataset.')
    parser.add_argument(
        '--sonar-dir', required=True, help='folder of sonar images')
    parser.add_argument(
        '--mask-dir', required=True, help='folder of binary masks')
    parser.add_argument(
        '--out-dir',
        default='data/pix2pix/sonar',
        help='output dataroot; train/, val/ and test/ are created inside it')
    parser.add_argument(
        '--size',
        type=int,
        default=256,
        help='each half is resized to size x size before concatenation')
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='fraction of pairs placed in the val split '
        '(used for evaluation during training)')
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='fraction of pairs placed in the held-out test split')
    parser.add_argument('--seed', type=int, default=0, help='shuffle seed')
    return parser.parse_args()


def index_by_stem(folder):
    """Map filename-stem -> full path for every image in a folder."""
    mapping = {}
    for path in sorted(glob(osp.join(folder, '*'))):
        if path.lower().endswith(IMG_EXTS):
            stem = osp.splitext(osp.basename(path))[0]
            mapping[stem] = path
    return mapping


def to_3ch(img):
    """Ensure a loaded image is 3-channel (replicate gray masks)."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 1:
        img = np.concatenate([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def main():
    args = parse_args()

    sonar = index_by_stem(args.sonar_dir)
    masks = index_by_stem(args.mask_dir)
    stems = sorted(set(sonar) & set(masks))

    missing_mask = sorted(set(sonar) - set(masks))
    missing_sonar = sorted(set(masks) - set(sonar))
    if missing_mask:
        print(f'[warn] {len(missing_mask)} sonar images without a mask, '
              f'e.g. {missing_mask[:5]}')
    if missing_sonar:
        print(f'[warn] {len(missing_sonar)} masks without a sonar image, '
              f'e.g. {missing_sonar[:5]}')
    if not stems:
        raise RuntimeError(
            'No matching stems between --sonar-dir and --mask-dir. '
            'Pairs are matched by filename (without extension).')

    random.seed(args.seed)
    random.shuffle(stems)
    n_test = int(round(len(stems) * args.test_ratio))
    n_val = int(round(len(stems) * args.val_ratio))
    split = {
        'test': stems[:n_test],
        'val': stems[n_test:n_test + n_val],
        'train': stems[n_test + n_val:],
    }

    size = (args.size, args.size)
    for phase, phase_stems in split.items():
        out_phase = osp.join(args.out_dir, phase)
        os.makedirs(out_phase, exist_ok=True)
        for stem in mmcv.track_iter_progress(phase_stems):
            sonar_img = to_3ch(mmcv.imread(sonar[stem], flag='color'))
            mask_img = to_3ch(mmcv.imread(masks[stem], flag='color'))
            sonar_img = mmcv.imresize(sonar_img, size)
            mask_img = mmcv.imresize(mask_img, size)
            # A|B = sonar | mask  (domain_a | domain_b)
            pair = np.concatenate([sonar_img, mask_img], axis=1)
            mmcv.imwrite(pair, osp.join(out_phase, f'{stem}.png'))

    print(f'\nDone. train={len(split["train"])}  val={len(split["val"])}  '
          f'test={len(split["test"])}')
    print(f'Set `fake_nums = {len(split["val"])}` in the pix2pix config '
          f'(val_evaluator runs on val/; keep test/ the same size, or set a '
          f'separate fake_nums for the test_evaluator).')


if __name__ == '__main__':
    main()
