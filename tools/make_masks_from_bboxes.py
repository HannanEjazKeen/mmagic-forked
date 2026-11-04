#!/usr/bin/env python3
"""Create binary masks for the marine-debris FLS sonar dataset.

The dataset ships a single ``annotations.json`` mapping each image filename to
a list of bounding boxes::

    {
      "marine-debris-aris3k-638.png": {
        "bounding-boxes": [
          {"class": "can", "width": 61, "height": 51,
           "top-left-x": 123, "top-left-y": 399, "class-id": 0}
        ]
      },
      ...
    }

For every sonar image this script writes a single-channel PNG mask of the same
size: pixels inside any annotated bounding box are 255 (foreground/object),
everything else is 0 (background). The mask filename matches the sonar image
filename, so the pair can be fed straight into tools/make_paired_sonar.py.

Example
-------
python tools/make_masks_from_bboxes.py \
    --images-dir /Users/hannanejazkeen/Downloads/marine-debris-watertank-release/fls-images \
    --ann /Users/hannanejazkeen/Downloads/marine-debris-watertank-release/fls-images/annotations.json \
    --out-dir /Users/hannanejazkeen/Downloads/marine-debris-watertank-release/masks
"""
import argparse
import json
import os
import os.path as osp
from glob import glob

from PIL import Image, ImageDraw

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def parse_args():
    p = argparse.ArgumentParser(
        description='Generate binary masks from bounding-box annotations.')
    p.add_argument(
        '--images-dir', required=True, help='folder of sonar images')
    p.add_argument(
        '--ann', required=True, help='path to annotations.json')
    p.add_argument(
        '--out-dir', required=True, help='output folder for mask PNGs')
    p.add_argument(
        '--include-unannotated',
        action='store_true',
        help='also write an all-black mask for images with no annotation '
        '(default: skip them)')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.ann) as f:
        ann = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    image_paths = sorted(
        p for p in glob(osp.join(args.images_dir, '*'))
        if p.lower().endswith(IMG_EXTS))

    n_written = 0
    n_boxes = 0
    n_skipped = 0
    for img_path in image_paths:
        fname = osp.basename(img_path)
        entry = ann.get(fname)
        boxes = entry.get('bounding-boxes', []) if entry else []

        if not boxes and not args.include_unannotated:
            n_skipped += 1
            continue

        with Image.open(img_path) as im:
            w, h = im.size  # PIL size is (width, height)

        mask = Image.new('L', (w, h), 0)  # single-channel, black
        draw = ImageDraw.Draw(mask)
        for b in boxes:
            x0 = max(0, int(b['top-left-x']))
            y0 = max(0, int(b['top-left-y']))
            x1 = min(w - 1, x0 + int(b['width']) - 1)
            y1 = min(h - 1, y0 + int(b['height']) - 1)
            if x1 >= x0 and y1 >= y0:
                draw.rectangle([x0, y0, x1, y1], fill=255)
                n_boxes += 1

        mask.save(osp.join(args.out_dir, fname))
        n_written += 1

    print(f'Wrote {n_written} masks ({n_boxes} boxes total) to {args.out_dir}')
    if not args.include_unannotated:
        print(f'Skipped {n_skipped} images with no annotation '
              f'(use --include-unannotated to emit empty masks for them)')


if __name__ == '__main__':
    main()
