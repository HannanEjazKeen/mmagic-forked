_base_ = [
    '../_base_/models/base_pix2pix.py',
    '../_base_/datasets/paired_imgs_256x256_crop.py',
    '../_base_/gen_default_runtime.py'
]

# ===========================================================================
# EXPERIMENT SELECTORS -- edit just these two to pick a run.
#   DATASET:       'marine_debris' | 'uatd'
#   DISCRIMINATOR: 'pixelgan' | 'patchgan16' | 'patchgan70' | 'imagegan'
# The 4 discriminators x 2 datasets give the 8 runs from the paper, each
# written to its own work_dir so they never overwrite each other.
# ===========================================================================
DATASET = 'marine_debris'
DISCRIMINATOR = 'patchgan70'
# ===========================================================================

# deterministic training can improve the performance of Pix2Pix
randomness = dict(deterministic=True)

# A = SONAR image (what we generate), B = binary mask (input).
# Each paired file is [sonar | mask] = [A | B] (concatenated along width).
source_domain = domain_b = 'mask'
target_domain = domain_a = 'sonar'

# ---- dataset table --------------------------------------------------------
# `dataroot` points at the `paired/` folder written by tools/make_paired_*.py.
# `fake_nums` MUST equal the number of images in the evaluated split (val/);
# set it to the count the pairing script printed. Keep val/ and test/ equal in
# size, or give val_evaluator / test_evaluator separate fake_nums.
_DATASETS = {
    'marine_debris': dict(
        dataroot='/mnt/data/water-data/underwater/marine-debris/'
        'watertank-segmentation/Processed/paired',
        fake_nums=187),  # TODO: set to your marine-debris val count
    'uatd': dict(
        dataroot='/mnt/data/water-data/underwater/UATD/Processed/paired',
        fake_nums=920),  # TODO: set to your UATD val count
}

# ---- discriminator table --------------------------------------------------
# Receptive field is set by the discriminator depth. For PatchDiscriminator in
# this repo, num_conv == pix2pix n_layers, and the field (on 256px input) is:
#   num_conv=1 -> 16x16,  num_conv=3 -> 70x70,  num_conv=5 -> 286x286 (>=image).
# PixelGAN (1x1) needs the dedicated 1x1 PixelDiscriminator (all stride-1
# kernel-1 convs); PatchDiscriminator cannot go below 16x16.
# in_channels=6 = concat(sonar 3ch, mask 3ch).
_disc_common = dict(
    in_channels=6, base_channels=64, norm_cfg=dict(type='BN'),
    init_cfg=dict(type='normal', gain=0.02))
_DISCRIMINATORS = {
    'pixelgan': dict(type='PixelDiscriminator', **_disc_common),
    'patchgan16': dict(type='PatchDiscriminator', num_conv=1, **_disc_common),
    'patchgan70': dict(type='PatchDiscriminator', num_conv=3, **_disc_common),
    'imagegan': dict(type='PatchDiscriminator', num_conv=5, **_disc_common),
}
# actual receptive field per variant, folded into the work_dir name so the
# folder itself records it (no extra logging needed).
_RECEPTIVE_FIELD = {
    'pixelgan': 1,
    'patchgan16': 16,
    'patchgan70': 70,
    'imagegan': 286,
}

assert DATASET in _DATASETS, f'unknown DATASET {DATASET!r}'
assert DISCRIMINATOR in _DISCRIMINATORS, \
    f'unknown DISCRIMINATOR {DISCRIMINATOR!r}'

dataroot = _DATASETS[DATASET]['dataroot']
fake_nums = _DATASETS[DATASET]['fake_nums']

# model settings: only the discriminator changes across the 4 variants.
model = dict(
    default_domain=target_domain,
    reachable_domains=[target_domain],
    related_domains=[target_domain, source_domain],
    discriminator=_DISCRIMINATORS[DISCRIMINATOR],
    data_preprocessor=dict(data_keys=[f'img_{domain_a}', f'img_{domain_b}']))

max_iters = 80000
train_cfg = dict(max_iters=max_iters)

# pix2pix LR schedule: hold lr constant for the first half, then linearly
# decay to 0 over the second half (applied to both optimizers). This is the
# standard recipe and stabilises the late iterations where loss plateaued.
param_scheduler = dict(
    type='LinearLrInterval',
    interval=400,
    by_epoch=False,
    start_factor=2e-4,
    end_factor=0,
    begin=max_iters // 2,
    end=max_iters)

# keep each of the 8 runs in its own directory; the folder name carries the
# dataset, discriminator and its receptive field (e.g. ..._patchgan70_rf70).
work_dir = (f'./work_dirs/pix2pix_{DATASET}_{DISCRIMINATOR}'
            f'_rf{_RECEPTIVE_FIELD[DISCRIMINATOR]}')

# ---- pipeline wiring ------------------------------------------------------
train_pipeline = _base_.train_dataloader.dataset.pipeline
val_pipeline = _base_.val_dataloader.dataset.pipeline
test_pipeline = _base_.test_dataloader.dataset.pipeline

key_mapping = dict(
    type='KeyMapper',
    mapping={
        f'img_{domain_a}': 'img_A',
        f'img_{domain_b}': 'img_B'
    },
    remapping={
        f'img_{domain_a}': f'img_{domain_a}',
        f'img_{domain_b}': f'img_{domain_b}'
    })
pack_input = dict(
    type='PackInputs',
    keys=[f'img_{domain_a}', f'img_{domain_b}'],
    data_keys=[f'img_{domain_a}', f'img_{domain_b}'])

train_pipeline += [key_mapping, pack_input]
val_pipeline += [key_mapping, pack_input]
test_pipeline += [key_mapping, pack_input]

# batch_size=16 to use the reserved GPU (the bs=4 run used ~1.3 GB) and give
# BatchNorm steadier statistics. num_workers=1 matches a single-CPU Slurm
# allocation; data loading was not a bottleneck (data_time ~0.01s).
train_dataloader = dict(
    batch_size=16, num_workers=1, dataset=dict(data_root=dataroot))
# val/ is read during training by the eval hook (model selection).
val_dataloader = dict(
    num_workers=1,
    dataset=dict(data_root=dataroot, test_mode=True, test_dir='val'))
# test/ is the held-out split used by tools/test.py.
test_dataloader = dict(
    num_workers=1,
    dataset=dict(data_root=dataroot, test_mode=True, test_dir='test'))

# optimizer
optim_wrapper = dict(
    generators=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999))),
    discriminators=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999))))

custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=5000,
        fixed_input=True,
        vis_kwargs_list=[
            dict(type='Translation', name='trans'),
            dict(type='TranslationVal', name='trans_val')
        ])
]

# save the best checkpoint by FID
default_hooks = dict(checkpoint=dict(save_best='FID-Full/fid', rule='less'))

metrics = [
    dict(
        type='TransIS',
        prefix='IS-Full',
        fake_nums=fake_nums,
        inception_style='PyTorch',
        fake_key=f'fake_{target_domain}',
        sample_model='orig'),
    dict(
        type='TransFID',
        prefix='FID-Full',
        fake_nums=fake_nums,
        inception_style='PyTorch',
        real_key=f'img_{target_domain}',
        fake_key=f'fake_{target_domain}',
        sample_model='orig')
]

val_evaluator = dict(metrics=metrics)
test_evaluator = dict(metrics=metrics)

# real-time visualization: add TensorBoard alongside the default local backend.
# View live loss/FID/IS curves and sample images with:
#   tensorboard --logdir work_dirs
vis_backends = [dict(type='VisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='Visualizer', vis_backends=vis_backends)
