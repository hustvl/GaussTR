import torch
import torch.nn as nn
from mmdet3d.registry import MODELS

from .utils import (apply_to_items, generate_grid, get_covariance,
                    quat_to_rotmat)


def splat_into_3d(grid_coords,
                  means3d,
                  opacities,
                  covariances,
                  features=None,
                  eps=1e-6):
    grid_coords = grid_coords.reshape(1, -1, 3).repeat(
        (means3d.size(0), 1, 1))  # (bs, grids, 3)
    grid_density = torch.zeros((*grid_coords.shape[:2], 1),
                               device=grid_coords.device)
    if features is not None:
        grid_feats = torch.zeros((*grid_coords.shape[:2], features.size(-1)),
                                 device=grid_coords.device)
    grid_cnts = torch.zeros_like(grid_density, dtype=torch.int)

    for g in range(means3d.size(1)):
        diff = grid_coords - means3d[:, g, None]
        maha_dist = diff.unsqueeze(2) @ covariances[:, g, None].inverse()
        maha_dist = (maha_dist @ diff.unsqueeze(-1)).squeeze(-1)

        mask = maha_dist <= 7.5
        grid_cnts += mask
        denom = grid_cnts.clamp(1e-6)
        density = opacities[:, g, None] * torch.exp(-0.5 * maha_dist)
        grid_density += mask * (density - grid_density) / denom

        if features is None:
            continue
        # NOTE: w/o dist. decay `* torch.exp(-0.5 * maha_dist)`
        feats = opacities[:, g, None] * features[:, g, None]
        grid_feats += mask * (feats - grid_feats) / denom
    return (grid_density, grid_feats) if features is not None else grid_density


@MODELS.register_module()
class GaussianVoxelizer(nn.Module):

    def __init__(self,
                 vol_range,
                 voxel_size,
                 filter_gaussians=False,
                 opacity_thresh=0):
        super().__init__()
        self.vol_range = vol_range
        self.grid_shape = [
            int((vol_range[i + 3] - vol_range[i]) / voxel_size)
            for i in range(3)
        ]
        grid_coords = generate_grid(self.grid_shape, offset=0.5)
        grid_coords = grid_coords * voxel_size + torch.tensor(vol_range[:3])
        self.register_buffer('grid_coords', grid_coords)

        self.filter_gaussians = filter_gaussians
        self.opacity_thresh = opacity_thresh

    def forward(self, **gaussians):
        """
        gaussians:
            means3d, opacities, features, covariances / (scales, rotations)
        """
        if self.filter_gaussians:
            assert gaussians['means3d'].size(0) == 1
            mask = gaussians['opacities'][0, :, 0] > self.opacity_thresh
            for i in range(3):
                mask &= (
                    gaussians['means3d'][0, :, i] >= self.vol_range[i]) & (
                        gaussians['means3d'][0, :, i] <= self.vol_range[i + 3])
            gaussians = apply_to_items(lambda x: x[:, mask], gaussians)

        if 'covariances' not in gaussians:
            gaussians['covariances'] = get_covariance(
                gaussians.pop('scales'),
                quat_to_rotmat(gaussians.pop('rotations')))
        outs = splat_into_3d(self.grid_coords, **gaussians)
        outs = [
            out.reshape(out.size(0), *self.grid_shape, out.size(-1))
            for out in (outs if isinstance(outs, tuple) else [outs])
        ]
        return outs if len(outs) > 1 else outs[0]
