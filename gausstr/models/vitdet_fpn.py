import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS


@MODELS.register_module()
class LN2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs
    pointwise mean and variance normalization over the channel dimension for
    inputs that have shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def get_scaling_modules(scale, channels, norm_cfg):
    assert -2 <= scale <= 1
    match scale:
        case -2:
            return nn.Sequential(
                nn.ConvTranspose2d(channels, channels // 2, 2, 2),
                build_norm_layer(norm_cfg, channels // 2)[1], nn.GELU(),
                nn.ConvTranspose2d(channels // 2, channels // 4, 2, 2))
        case -1:
            return nn.ConvTranspose2d(channels, channels // 2, 2, 2)
        case 0:
            return nn.Identity()
        case 1:
            return nn.MaxPool2d(kernel_size=2, stride=2)


@MODELS.register_module()
class ViTDetFPN(BaseModule):
    """Simple Feature Pyramid Network for ViTDet."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 scales=(-2, -1, 0, 1),
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.scale_convs = nn.ModuleList([
            get_scaling_modules(scale, in_channels, norm_cfg)
            for scale in scales
        ])
        channels = [int(in_channels * 2**min(scale, 0)) for scale in scales]

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(len(channels)):
            l_conv = ConvModule(
                channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, x):
        inputs = [scale_conv(x) for scale_conv in self.scale_convs]
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        outs = [
            fpn_conv(laterals[i]) for i, fpn_conv in enumerate(self.fpn_convs)
        ]
        return outs
