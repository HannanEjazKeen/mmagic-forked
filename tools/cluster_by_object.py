# Copyright (c) OpenMMLab. All rights reserved.
"""Cluster full sonar images by the objects they contain, via template match.

The dataset has two folders:
  * ``--seg-dir``:  full sonar images (watertank-segmentation/Images);
  * ``--crop-dir``: one sub-folder per object class (tire/, tile/, ...), each
    holding crops of that object taken from the full images.

Because every crop is a sub-region cut from some full image, we can find which
full images contain an object by sliding each crop over each full image with
normalized cross-correlation (cv2.matchTemplate, TM_CCOEFF_NORMED). A match
score >= --threshold means "this object appears in this image". Each full
image is then linked into ``--out-dir/<object>/`` for every object it contains
(an image with several objects appears under several folders), and a
``image_objects.csv`` records the full image -> objects mapping.

Speed: for each object we skip images already matched to it, so once an object
has been located in a frame we stop testing more crops of it against that
frame -- a big saving when objects recur across near-duplicate video frames.
Use --scale 0.5 to match on half-resolution for a ~4x speedup.

Example
-------
python tools/cluster_by_object.py \
    --seg-dir  /mnt/data/.../watertank-segmentation/Images \
    --crop-dir /mnt/data/.../watertank-cropped \
    --out-dir  /mnt/data/.../watertank-by-object \
    --threshold 0.9 --link symlink
"""
import argparse
import csv
import os
import os.path as osp
from collections import defaultdict
from glob import glob

import cv2
from mmengine.utils import track_iter_progress

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Cluster sonar images by object via template matching.')
    parser.add_argument('--seg-dir', required=True, help='full sonar images')
    parser.add_argument(
        '--crop-dir', required=True,
        help='root with one sub-folder of crops per object class')
    parser.add_argument('--out-dir', default='data/by_object')
    parser.add_argument(
        '--threshold', type=float, default=0.9,
        help='TM_CCOEFF_NORMED match score to count an object as present')
    parser.add_argument(
        '--scale', type=float, default=1.0,
        help='match at this resolution factor (0.5 = half res, ~4x faster)')
    parser.add_argument(
        '--link', choices=['symlink', 'copy', 'none'], default='symlink',
        help='how to populate out-dir/<object>/ (none = only write the CSV)')
    return parser.parse_args()


def list_images(folder):
    return sorted(
        p for p in glob(osp.join(folder, '*'))
        if p.lower().endswith(IMG_EXTS))


def load_gray(path, scale):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if scale != 1.0:
        img = cv2.resize(
            img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def main():
    args = parse_args()

    seg_paths = list_images(args.seg_dir)
    if not seg_paths:
        raise RuntimeError(f'No images in --seg-dir {args.seg_dir}')
    seg = {}
    print(f'Loading {len(seg_paths)} full images...')
    for p in track_iter_progress(seg_paths):
        g = load_gray(p, args.scale)
        if g is not None:
            seg[p] = g

    # object class -> list of crop paths
    objects = {}
    for d in sorted(glob(osp.join(args.crop_dir, '*'))):
        if osp.isdir(d):
            crops = list_images(d)
            if crops:
                objects[osp.basename(d)] = crops
    if not objects:
        raise RuntimeError(f'No object sub-folders with crops in {args.crop_dir}')
    print(f'Objects: {", ".join(f"{k}({len(v)})" for k, v in objects.items())}')

    # image path -> set of object classes present
    image_objects = defaultdict(set)

    for obj, crops in objects.items():
        matched = set()  # seg paths already known to contain this object
        print(f'\n[{obj}] matching {len(crops)} crops against '
              f'{len(seg)} images...')
        for crop_path in track_iter_progress(crops):
            tmpl = load_gray(crop_path, args.scale)
            if tmpl is None:
                continue
            th, tw = tmpl.shape[:2]
            for spath, simg in seg.items():
                if spath in matched:
                    continue  # already found this object here
                if th > simg.shape[0] or tw > simg.shape[1]:
                    continue  # template bigger than image
                res = cv2.matchTemplate(simg, tmpl, cv2.TM_CCOEFF_NORMED)
                if res.max() >= args.threshold:
                    matched.add(spath)
        for spath in matched:
            image_objects[spath].add(obj)
        print(f'  -> {obj}: found in {len(matched)} images')

    # write outputs
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = osp.join(args.out_dir, 'image_objects.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'objects'])
        for spath in seg_paths:
            objs = sorted(image_objects.get(spath, []))
            w.writerow([osp.basename(spath), ';'.join(objs)])

    if args.link != 'none':
        for obj in objects:
            os.makedirs(osp.join(args.out_dir, obj), exist_ok=True)
        for spath, objs in image_objects.items():
            for obj in objs:
                dst = osp.join(args.out_dir, obj, osp.basename(spath))
                if osp.lexists(dst):
                    continue
                if args.link == 'symlink':
                    os.symlink(osp.abspath(spath), dst)
                else:
                    import shutil
                    shutil.copy2(spath, dst)

    n_labeled = sum(1 for p in seg_paths if image_objects.get(p))
    print(f'\nDone. {n_labeled}/{len(seg_paths)} images matched >=1 object.')
    print(f'Mapping: {csv_path}')
    if args.link != 'none':
        print(f'Per-object folders ({args.link}) under: {args.out_dir}')


if __name__ == '__main__':
    main()
