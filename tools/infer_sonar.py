# Copyright (c) OpenMMLab. All rights reserved.
"""Standalone pix2pix sonar inference (mask -> sonar).

Generates sonar images from the mask half of the paired ``[sonar | mask]``
test images, using a trained checkpoint. It builds the model straight from the
config + checkpoint via mmengine and the model registry, deliberately avoiding
``mmagic.apis`` (whose inferencer bundle imports controlnet_aux / mediapipe /
skimage and breaks on partial installs).

For every ``*.png`` in ``--pair-dir`` (the paired test folder) it writes:
  * ``<out>/gen/<stem>.png``     -- the generated sonar image;
  * ``<out>/compare/<stem>.png`` -- [input mask | generated | real sonar],
    for side-by-side visual QA.

Example
-------
python tools/infer_sonar.py \
    configs/pix2pix/pix2pix_sonar_disc-variants.py \
    work_dirs/pix2pix_marine_debris_patchgan70_rf70/best_FID-Full_fid_iter_30000.pth \
    --pair-dir /mnt/data/.../Processed/paired/test \
    --out-dir ./test_out
"""
import argparse
import os
import os.path as osp
from glob import glob

import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.dataset.utils import default_collate as collate
from mmengine.runner import load_checkpoint
from torchvision import utils

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def parse_args():
    parser = argparse.ArgumentParser(description='pix2pix sonar inference')
    parser.add_argument('config', help='model config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--pair-dir',
        required=True,
        help='folder of paired [sonar|mask] images (e.g. <dataroot>/test)')
    parser.add_argument(
        '--out-dir', default='./test_out', help='output directory')
    parser.add_argument('--device', default='cuda:0', help='inference device')
    return parser.parse_args()


def to01(img):
    """[-1,1] BGR tensor -> [0,1] RGB tensor for saving."""
    return ((img[:, [2, 1, 0]] + 1.0) / 2.0).clamp(0, 1)


def main():
    args = parse_args()
    register_all_modules()

    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(args.device)
    model.eval()

    # mask -> sonar: target is the default domain, source is the other one.
    target_domain = model._default_domain
    source_domain = model.get_other_domains(target_domain)[0]

    # reuse the exact test-time pipeline (splits the [A|B] pair, resizes,
    # packs) and the model's own data_preprocessor (normalisation).
    test_pipeline = Compose(cfg.test_pipeline)

    gen_dir = osp.join(args.out_dir, 'gen')
    cmp_dir = osp.join(args.out_dir, 'compare')
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(cmp_dir, exist_ok=True)

    pairs = sorted(
        p for p in glob(osp.join(args.pair_dir, '*'))
        if p.lower().endswith(IMG_EXTS))
    if not pairs:
        raise RuntimeError(f'No images found in {args.pair_dir}')
    print(f'Found {len(pairs)} paired images. Writing to {args.out_dir}')

    for path in pairs:
        stem = osp.splitext(osp.basename(path))[0]
        # the paired loader keys everything off a single path
        data = dict(pair_path=path, img_A_path=path, img_B_path=path)
        data = collate([test_pipeline(data)])
        data = model.data_preprocessor(data, False)
        inputs = data['inputs']

        with torch.no_grad():
            results = model(
                inputs[f'img_{source_domain}'].to(args.device),
                test_mode=True,
                target_domain=target_domain)
        gen = results['target'].cpu()

        utils.save_image(to01(gen), osp.join(gen_dir, f'{stem}.png'))
        # [ mask | generated | real sonar ]
        row = torch.cat([
            to01(inputs[f'img_{source_domain}'].cpu()),
            to01(gen),
            to01(inputs[f'img_{target_domain}'].cpu()),
        ], dim=3)
        utils.save_image(row, osp.join(cmp_dir, f'{stem}.png'))

    print(f'Done. Generated images: {gen_dir}\nComparisons: {cmp_dir}')


if __name__ == '__main__':
    main()
