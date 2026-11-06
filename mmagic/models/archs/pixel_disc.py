# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch.nn as nn
from mmcv.cnn import ConvModule, build_conv_layer
from mmengine.model import BaseModule
from torch import Tensor

from mmagic.models.utils import generation_init_weights
from mmagic.registry import MODELS


@MODELS.register_module()
class PixelDiscriminator(BaseModule):
    """A 1x1 PatchGAN discriminator (a.k.a. PixelGAN).

    Every convolution is 1x1 with stride 1, so the receptive field is a single
    pixel: the discriminator judges each pixel independently. This is the
    smallest-receptive-field variant from Isola et al. (pix2pix), used as the
    ``PixelGAN`` configuration. For larger receptive fields
    (16x16 / 70x70 / 286x286) use :class:`PatchDiscriminator` with
    ``num_conv`` set to 1 / 3 / 5 respectively.

    Args:
        in_channels (int): Number of channels in input images.
        base_channels (int): Number of channels at the first conv layer.
            Default: 64.
        norm_cfg (dict): Config dict to build norm layer. Default:
            `dict(type='BN')`.
        init_cfg (dict): Config dict for initialization.
            `type`: The name of our initialization method. Default: 'normal'.
            `gain`: Scaling factor for normal, xavier and orthogonal.
            Default: 0.02.
    """

    def __init__(self,
                 in_channels: int,
                 base_channels: int = 64,
                 norm_cfg: dict = dict(type='BN'),
                 init_cfg: Optional[dict] = dict(type='normal', gain=0.02)):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        # Only for IN, use bias since it does not have affine parameters.
        use_bias = norm_cfg['type'] == 'IN'

        sequence = [
            ConvModule(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                norm_cfg=None,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.2)),
            ConvModule(
                in_channels=base_channels,
                out_channels=base_channels * 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=use_bias,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='LeakyReLU', negative_slope=0.2)),
            # output one-channel prediction map
            build_conv_layer(
                dict(type='Conv2d'),
                base_channels * 2,
                1,
                kernel_size=1,
                stride=1,
                padding=0)
        ]

        self.model = nn.Sequential(*sequence)
        self.init_type = 'normal' if init_cfg is None else init_cfg.get(
            'type', 'normal')
        self.init_gain = 0.02 if init_cfg is None else init_cfg.get(
            'gain', 0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        return self.model(x)

    def init_weights(self) -> None:
        """Initialize weights for the model."""
        if self.init_cfg is None and self.init_cfg['type'] == 'Pretrained':
            super().init_weights()
            return
        generation_init_weights(
            self, init_type=self.init_type, init_gain=self.init_gain)
        self._is_init = True
