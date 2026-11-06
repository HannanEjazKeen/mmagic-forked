# Copyright (c) OpenMMLab. All rights reserved.
"""Build a sonar pix2pix dataset from separate sonar images and binary masks.

Input: one folder of sonar images and one folder of binary masks, matched by
filename stem (``sonar/0001.png`` <-> ``masks/0001.png``). The matched pairs
are shuffled and split into train / val / test by ratio.

Output (see tools/sonar_dataset_utils.py for the full layout):
  * unpaired ``<out>/{train,val,test}/{images,masks}`` - reusable by other
    models (segmentation, detection, ...);
  * paired ``<out>/paired/{train,val,test}/*.png``, each ``[sonar | mask]``
    (A|B), which is what the pix2pix config consumes.

Point the pix2pix config's ``dataroot`` at ``<out>/paired``.

Example
-------
python tools/make_paired_sonar.py \
    --sonar-dir /path/to/sonar_images \
    --mask-dir  /path/to/binary_masks \
    --out-dir   data/pix2pix/sonar \
    --size 256 --val-ratio 0.1 --test-ratio 0.1 --seed 0
"""
import argparse

import mmcv

from sonar_dataset_utils import (build_dataset, index_by_stem, report,
                                  split_stems, to_3ch)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Make a sonar pix2pix [sonar|mask] paired dataset '
        'from flat images/ + masks/ folders.')
    parser.add_argument(
        '--sonar-dir', required=True, help='folder of sonar images')
    parser.add_argument(
        '--mask-dir', required=True, help='folder of binary masks')
    parser.add_argument(
        '--out-dir',
        default='data/pix2pix/sonar',
        help='output dataroot; unpaired splits and a paired/ folder are '
        'created inside it')
    parser.add_argument(
        '--size',
        type=int,
        default=256,
        help='each half is resized to size x size')
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='fraction of pairs placed in the val split')
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='fraction of pairs placed in the held-out test split')
    parser.add_argument('--seed', type=int, default=0, help='shuffle seed')
    return parser.parse_args()


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

    def loader(stem):
        sonar_img = to_3ch(mmcv.imread(sonar[stem], flag='color'))
        mask_img = to_3ch(mmcv.imread(masks[stem], flag='color'))
        return sonar_img, mask_img

    split = split_stems(stems, args.val_ratio, args.test_ratio, args.seed)
    build_dataset(args.out_dir, split, loader, args.size)
    report(split, args.out_dir)


if __name__ == '__main__':
    main()
