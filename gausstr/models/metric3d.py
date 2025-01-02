# Referred to https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class Metric3D(nn.Module):

    def __init__(self, model_name='metric3d_vit_large'):
        super().__init__()
        self.model = torch.hub.load(
            'yvanyin/metric3d', model_name, pretrain=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.input_size = (616, 1064)
        self.canonical_focal = 1000.0

    @torch.no_grad()
    def forward(self, x, cam2img, img_aug_mat=None):
        bs, n, _, *ori_shape = x.shape
        x = x.flatten(0, 1)
        scale = min(self.input_size[0] / ori_shape[0],
                    self.input_size[1] / ori_shape[1])
        x = F.interpolate(x, scale_factor=scale, mode='bilinear')

        h, w = x.shape[-2:]
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
        pred_depth = pred_depth.reshape(bs, n, 1, *pred_depth.shape[-2:])

        canonical_to_real_scale = (
            cam2img[..., 0, 0] * scale / self.canonical_focal)
        if img_aug_mat is not None:
            canonical_to_real_scale *= img_aug_mat[..., 0, 0]
        canonical_to_real_scale = canonical_to_real_scale.reshape(
            bs, n, 1, 1, 1)
        return pred_depth * canonical_to_real_scale

    def visualize(self, images, depths):
        images = images.permute(0, 2, 3, 1).flatten(0, 1).cpu().numpy()
        depths = depths.flatten(0, 2).cpu().numpy()
        images = (images - images.min()) / (images.max() - images.min())

        fig, axes = plt.subplots(1, 2, figsize=(10, 1 * 5))
        axes[0].imshow(images)
        axes[0].axis('off')
        im = axes[1].imshow(depths, cmap='plasma')
        axes[1].axis('off')
        fig.colorbar(im, ax=axes[1])

        plt.tight_layout()
        plt.show()
