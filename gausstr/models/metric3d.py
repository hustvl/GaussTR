# Referred to https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class Metric3D(nn.Module):

    def __init__(self, model_name='metric3d_vit_large'):
        super().__init__()
        self.model = torch.hub.load(
            'yvanyin/metric3d', model_name, pretrain=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.input_size = (616, 1064)
        self.canonical_focal = 1000.0

    def forward(self, x, cam2img, img_aug_mat=None):
        ori_shape = x.shape[2:]
        scale = min(self.input_size[0] / ori_shape[0],
                    self.input_size[1] / ori_shape[1])
        x = F.interpolate(x, scale_factor=scale, mode='bilinear')

        h, w = x.shape[2:]
        pad_h = self.input_size[0] - h
        pad_w = self.input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        pad_info = [
            pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half
        ]
        x = F.pad(x, pad_info[2:] + pad_info[:2])

        if self.model.training:
            self.model.eval()
        pred_depth = self.model.inference({'input': x})[0]

        pred_depth = pred_depth[...,
                                pad_info[0]:pred_depth.shape[2] - pad_info[1],
                                pad_info[2]:pred_depth.shape[3] - pad_info[3]]
        pred_depth = F.interpolate(pred_depth, ori_shape, mode='bilinear')

        canonical_to_real = (cam2img[:, 0, 0] * scale / self.canonical_focal)
        if img_aug_mat is not None:
            canonical_to_real *= img_aug_mat[:, 0, 0]
        return pred_depth.squeeze(1) * canonical_to_real.reshape(-1, 1, 1)

    def visualize(self, x):
        x = x.cpu().numpy()
        x = (x - x.min()) / (x.max() - x.min())
        if x.ndim == 2:
            cmap = plt.get_cmap('Spectral_r')
            x = cmap(x)[..., :3]
        else:
            x = x.transpose(1, 2, 0)
        plt.imsave('metric3d_vis.png', x)
