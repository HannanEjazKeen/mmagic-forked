# Copyright (c) OpenMMLab. All rights reserved.
"""Build a sonar pix2pix dataset from the UATD dataset.

UATD ships pre-split folders (``train``, ``test1``, ``test2``), each with an
``images`` folder and an annotation folder of Pascal-VOC XML files (bounding
boxes). There are no masks, so a *binary* mask is generated per image by
filling every ``<bndbox>`` white on a black background -- i.e. every annotated
box is treated as one "object" class (no per-class distinction).

All three UATD folders are pooled and re-split into train / val / test by
ratio. Output layout is identical to make_paired_sonar.py (see
tools/sonar_dataset_utils.py): unpaired ``<out>/{train,val,test}/{images,
masks}`` plus paired ``<out>/paired/{train,val,test}/*.png`` as ``[sonar|mask]``.

Point the pix2pix config's ``dataroot`` at ``<out>/paired``.

Example
-------
python tools/make_paired_uatd.py \
    --uatd-root /path/to/UATD \
    --subsets train test1 test2 \
    --images-name images --anno-name annotations \
    --out-dir data/pix2pix/uatd \
    --size 256 --val-ratio 0.1 --test-ratio 0.1 --seed 0
"""
import argparse
import os.path as osp
import xml.etree.ElementTree as ET
from glob import glob

import mmcv
import numpy as np

from sonar_dataset_utils import (build_dataset, index_by_stem, report,
                                  split_stems, to_3ch)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Make a sonar pix2pix [sonar|mask] dataset from UATD '
        '(Pascal-VOC bounding boxes -> binary masks).')
    parser.add_argument(
        '--uatd-root', required=True, help='root containing the UATD subsets')
    parser.add_argument(
        '--subsets',
        nargs='+',
        default=['train', 'test1', 'test2'],
        help='subset folders under --uatd-root to pool together')
    parser.add_argument(
        '--images-name',
        default='images',
        help='name of the images sub-folder inside each subset')
    parser.add_argument(
        '--anno-name',
        default='annotations',
        help='name of the XML annotation sub-folder inside each subset')
    parser.add_argument(
        '--out-dir',
        default='data/pix2pix/uatd',
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


def index_xml_by_stem(folder):
    """Map filename-stem -> full path for every .xml in a folder."""
    mapping = {}
    for path in sorted(glob(osp.join(folder, '*.xml'))):
        stem = osp.splitext(osp.basename(path))[0]
        mapping[stem] = path
    return mapping


def mask_from_voc(xml_path, img_h, img_w):
    """Rasterize every <bndbox> in a Pascal-VOC XML as a filled white box.

    Falls back to the image size when the XML has no <size>. Boxes are clipped
    to the image bounds. Returns a 3-channel uint8 mask (0 or 255).
    """
    root = ET.parse(xml_path).getroot()

    # Prefer the size declared in the XML; scale boxes if it differs from the
    # actual image (defensive -- usually they match).
    size = root.find('size')
    xml_w = int(float(size.find('width').text)) if size is not None else img_w
    xml_h = int(float(size.find('height').text)) \
        if size is not None else img_h
    sx = img_w / xml_w if xml_w else 1.0
    sy = img_h / xml_h if xml_h else 1.0

    mask = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for obj in root.findall('object'):
        box = obj.find('bndbox')
        if box is None:
            continue
        xmin = int(round(float(box.find('xmin').text) * sx))
        ymin = int(round(float(box.find('ymin').text) * sy))
        xmax = int(round(float(box.find('xmax').text) * sx))
        ymax = int(round(float(box.find('ymax').text) * sy))
        xmin, xmax = sorted((max(0, xmin), min(img_w, xmax)))
        ymin, ymax = sorted((max(0, ymin), min(img_h, ymax)))
        mask[ymin:ymax, xmin:xmax] = 255
    return mask


def main():
    args = parse_args()

    # Pool all subsets, matching each image stem to its XML. Stems are
    # namespaced by subset so identical names across subsets don't collide.
    img_paths, xml_paths = {}, {}
    for subset in args.subsets:
        img_dir = osp.join(args.uatd_root, subset, args.images_name)
        anno_dir = osp.join(args.uatd_root, subset, args.anno_name)
        if not osp.isdir(img_dir):
            raise RuntimeError(f'Images folder not found: {img_dir}')
        if not osp.isdir(anno_dir):
            raise RuntimeError(f'Annotation folder not found: {anno_dir}')
        imgs = index_by_stem(img_dir)
        xmls = index_xml_by_stem(anno_dir)
        matched = sorted(set(imgs) & set(xmls))
        missing = sorted(set(imgs) - set(xmls))
        if missing:
            print(f'[warn] {subset}: {len(missing)} images without XML, '
                  f'e.g. {missing[:5]}')
        for stem in matched:
            key = f'{subset}/{stem}'
            img_paths[key] = imgs[stem]
            xml_paths[key] = xmls[stem]

    stems = sorted(img_paths)
    if not stems:
        raise RuntimeError('No image/XML pairs found across the given '
                           '--subsets. Check --images-name / --anno-name.')
    print(f'Pooled {len(stems)} image/XML pairs from '
          f'{len(args.subsets)} subset(s).')

    def loader(key):
        sonar_img = to_3ch(mmcv.imread(img_paths[key], flag='color'))
        h, w = sonar_img.shape[:2]
        mask_img = mask_from_voc(xml_paths[key], h, w)
        return sonar_img, mask_img

    split = split_stems(stems, args.val_ratio, args.test_ratio, args.seed)
    # Flatten "subset/stem" keys to safe filenames for output.
    flat = {
        phase: [k.replace('/', '__') for k in keys]
        for phase, keys in split.items()
    }
    key_of = {k.replace('/', '__'): k for k in stems}
    build_dataset(args.out_dir, flat, lambda s: loader(key_of[s]), args.size)
    report(split, args.out_dir)


if __name__ == '__main__':
    main()
