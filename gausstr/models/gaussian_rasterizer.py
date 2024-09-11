import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_gaussian_rasterization import (GaussianRasterizationSettings,
                                         rasterize_gaussians)
from torch.cuda.amp import autocast

from mmdet3d.registry import MODELS

from .utils import unbatched_forward


@MODELS.register_module()
class GaussianRasterizer(nn.Module):

    @unbatched_forward
    def forward(self,
                means3d,
                colors,
                opacities,
                cam2img,
                cam2ego,
                img_shape,
                scales=None,
                rotations=None,
                covariances=None,
                img_aug_mat=None):
        assert ((scales is not None and rotations is not None) ^
                (covariances is not None))
        if scales is not None and scales.size(-1) == 1:
            scales = scales.expand(-1, 3)
        # elif self.default_scale is not None:
        #     scales = torch.ones_like(means3d) * self.default_scale
        if covariances is None and rotations is None:
            rotations = torch.zeros((len(means3d), 4)).to(means3d)
            rotations[:, 0] = 1

        if hasattr(self, 'bg_color'):
            bg_color = self.bg_color.weight[0] if isinstance(
                self.bg_color, nn.Embedding) else self.bg_color
            if self.random_bg_color:
                bg_color *= random.choice([1, -1])
        else:
            bg_color = torch.zeros(colors.size(1)).to(colors.device)

        rendered_feats = []
        depths = []
        for v in range(cam2img.size(0)):
            cam2img_, img_shape_, crop = self.pad_before_splatting(
                cam2img[v], img_shape,
                img_aug_mat[v] if img_aug_mat is not None else None)
            rendered, depth = self.render_gaussian_splatting(
                means3d, colors, opacities.float(), scales, rotations,
                covariances, cam2img_, cam2ego[v], img_shape_, bg_color)

            rendered_feats.append(self.crop_after_splatting(rendered, crop))
            depths.append(self.crop_after_splatting(depth, crop))
        return torch.stack(rendered_feats), torch.stack(depths)

    def pad_before_splatting(self, cam2img, img_shape, img_aug_mat=None):
        cam2img = cam2img.clone()
        cx, cy = cam2img[0, 2].item(), cam2img[1, 2].item()
        h, w = img_shape

        if img_aug_mat is not None:
            cx = cx * img_aug_mat[0, 0] + img_aug_mat[0, 3]
            cy = cy * img_aug_mat[1, 1] + img_aug_mat[1, 3]
            w = w * img_aug_mat[0, 0] + img_aug_mat[0, 3]
            h = h * img_aug_mat[1, 1] + img_aug_mat[1, 3]
            cam2img[:2, :2] *= img_aug_mat[:2, :2]

        crop_l = int(max(w - cx * 2, 0))
        crop_r = int(max(cx * 2 - w, 0))
        crop_t = int(max(h - cy * 2, 0))
        crop_b = int(max(cy * 2 - h, 0))

        cx = max(cx, w - cx)
        cy = max(cy, h - cy)
        cam2img[0, 2] = cx
        cam2img[1, 2] = cy

        img_shape = (int(cy * 2), int(cx * 2))
        crop = (crop_l, crop_r, crop_t, crop_b)
        return cam2img, img_shape, crop

    def crop_after_splatting(self, img, crop, img_aug_mat=None):
        h, w = img.shape[-2:]
        img = img[:, crop[2]:h - crop[3], crop[0]:w - crop[1]]

        if img_aug_mat is not None:
            h, w = img.shape[-2:]
            img = F.interpolate(
                img.unsqueeze(0),
                (int(h * img_aug_mat[1, 1]), int(w * img_aug_mat[0, 0])),
                mode='bilinear').squeeze(0)
            img = img[:, -int(img_aug_mat[1, -1]):, -int(img_aug_mat[0, -1]):]
        return img

    @autocast(enabled=False)
    def render_gaussian_splatting(self,
                                  means3d,
                                  colors,
                                  opacities,
                                  scales,
                                  rotations,
                                  covariances,
                                  cam2img,
                                  cam2ego,
                                  img_size,
                                  bg_color=None):
        fx = cam2img[0, 0]
        fy = cam2img[1, 1]
        cx = cam2img[0, 2]
        cy = cam2img[1, 2]

        h, w = img_size
        r, l, t, b = w - cx, -cx, cy, cy - h
        tanfovx = w / 2 / fx
        tanfovy = h / 2 / fy

        R = cam2ego[:3, :3].transpose(0, 1)
        T = -R @ cam2ego[:3, 3]
        viewmatrix = torch.eye(4).to(cam2ego)
        viewmatrix[:3, :3] = R
        viewmatrix[:3, 3] = T
        viewmatrix = viewmatrix.transpose(0, 1)
        camera_center = viewmatrix.inverse()[3, :3]

        n, f = 0.1, 100
        projmatrix = torch.zeros_like(viewmatrix)
        projmatrix[0, 0] = 2 * fx / (r - l)
        projmatrix[1, 1] = 2 * fy / (t - b)
        projmatrix[0, 2] = (r + l) / (r - l)
        projmatrix[1, 2] = (t + b) / (t - b)
        projmatrix[2, 2] = f / (f - n)
        projmatrix[2, 3] = -(f * n) / (f - n)
        projmatrix[3, 2] = 1
        projmatrix = viewmatrix @ projmatrix.transpose(0, 1)

        raster_settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            sh_degree=0,
            campos=camera_center,
            prefiltered=False,
            debug=False)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        means2d = torch.zeros_like(means3d, requires_grad=True).to(means3d)
        try:
            means2d.retain_grad()
        except:
            pass

        shs = torch.Tensor([])
        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if covariances is None:
            covariances = torch.Tensor([])

        # self.save_ply(means3d, colors, opacities, scales, rotations)
        # Invoke C++/CUDA rasterization routine
        rendered, radii, depth = rasterize_gaussians(
            means3d, means2d, shs, colors, opacities, scales, rotations,
            covariances, raster_settings)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return rendered, depth

    def save_ply(self, means3d, colors, opacity, scales, rotations):
        means3d = means3d.detach().cpu().numpy()
        normals = np.zeros_like(means3d)
        colors = (
            colors.detach().cpu().numpy()
            if colors.shape[-1] == 3 else np.random.rand(colors.shape[0], 3))
        opacities = opacity.detach().cpu().numpy()
        scales = np.log(scales.detach().cpu().numpy())
        rotations = rotations.detach().cpu().numpy()

        attributes = ('x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1',
                      'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2',
                      'rot_0', 'rot_1', 'rot_2', 'rot_3')
        dtypes = [(attr, 'f4') for attr in attributes]

        elements = np.empty(means3d.shape[0], dtype=dtypes)
        attributes = np.concatenate(
            [means3d, normals, colors, opacities, scales, rotations], axis=1)
        elements[:] = list(map(tuple, attributes))

        import time

        from plyfile import PlyData, PlyElement
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(f'{time.time() % 1e6:.2f}.ply')
