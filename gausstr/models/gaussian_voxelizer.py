import torch
import torch.nn as nn
from mmdet3d.registry import MODELS

from .utils import (apply_to_items, generate_grid, get_covariance,
                    quat_to_rotmat, unbatched_forward)


def splat_into_3d(grid_coords,
                  means3d,
                  opacities,
                  covariances,
                  vol_range,
                  voxel_size,
                  features=None,
                  eps=1e-6):
    grid_density = torch.zeros((*grid_coords.shape[:-1], 1),
                               device=grid_coords.device)
    if features is not None:
        grid_feats = torch.zeros((*grid_coords.shape[:-1], features.size(-1)),
                                 device=grid_coords.device)

    for g in range(means3d.size(0)):
        sigma = torch.sqrt(torch.diag(covariances[g]))
        factor = 3 * torch.tensor([-1, 1])[:, None].to(sigma)
        bounds = means3d[g, None] + factor * sigma[None]
        if not (((bounds > vol_range[None, :3]).max(0).values.min()) and
                ((bounds < vol_range[None, 3:]).max(0).values.min())):
            continue
        bounds = bounds.clamp(vol_range[:3], vol_range[3:])
        bounds = ((bounds - vol_range[:3]) / voxel_size).int().tolist()
        slices = tuple([slice(lo, hi + 1) for lo, hi in zip(*bounds)])

        diff = grid_coords[slices] - means3d[g]
        maha_dist = (diff.unsqueeze(-2) @ covariances[g].inverse()
                     @ diff.unsqueeze(-1)).squeeze(-1)
        density = opacities[g] * torch.exp(-0.5 * maha_dist)
        grid_density[slices] += density
        if features is not None:
            grid_feats[slices] += density * features[g]

    if features is None:
        return grid_density
    grid_feats /= grid_density.clamp(eps)
    return grid_density, grid_feats


@MODELS.register_module()
class GaussianVoxelizer(nn.Module):

    def __init__(self,
                 vol_range,
                 voxel_size,
                 filter_gaussians=False,
                 opacity_thresh=0,
                 covariance_thresh=0):
        super().__init__()
        self.voxel_size = voxel_size
        vol_range = torch.tensor(vol_range)
        self.register_buffer('vol_range', vol_range)

        self.grid_shape = ((vol_range[3:] - vol_range[:3]) /
                           voxel_size).int().tolist()
        grid_coords = generate_grid(self.grid_shape, offset=0.5)
        grid_coords = grid_coords * voxel_size + vol_range[:3]
        self.register_buffer('grid_coords', grid_coords)

        self.filter_gaussians = filter_gaussians
        self.opacity_thresh = opacity_thresh
        self.covariance_thresh = covariance_thresh

    @unbatched_forward
    def forward(self,
                means3d,
                opacities,
                covariances=None,
                scales=None,
                rotations=None,
                **kwargs):
        if covariances is None:
            covariances = get_covariance(scales, quat_to_rotmat(rotations))
        gaussians = dict(
            means3d=means3d,
            opacities=opacities,
            covariances=covariances,
            **kwargs)

        if self.filter_gaussians:
            mask = opacities.squeeze(1) > self.opacity_thresh
            for i in range(3):
                mask &= (means3d[:, i] >= self.vol_range[i]) & (
                    means3d[:, i] <= self.vol_range[i + 3])
            if self.covariance_thresh > 0:
                cov_diag = torch.diagonal(covariances, dim1=1, dim2=2)
                mask &= ((cov_diag.min(1)[0] * 6) > self.covariance_thresh)
            gaussians = apply_to_items(lambda x: x[mask], gaussians)

        return splat_into_3d(
            self.grid_coords,
            **gaussians,
            vol_range=self.vol_range,
            voxel_size=self.voxel_size)
