import torch
import torch.nn as nn

from mmdet3d.registry import MODELS

from .utils import generate_grid, get_covariance, quat_to_rotmat


def apply_to_items(func, iterable):
    if isinstance(iterable, list):
        return [func(i) for i in iterable]
    elif isinstance(iterable, dict):
        return {k: func(v) for k, v in iterable.items()}


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

        mask = maha_dist <= 4
        grid_cnts += mask
        denom = grid_cnts.clamp(1e-6)
        density = opacities[:, g, None] * torch.exp(-0.5 * maha_dist)
        grid_density += mask * (density - grid_density) / denom

        if features is None:
            continue
        feats = opacities[:, g, None] * features[:, g, None] * torch.exp(
            -0.5 * maha_dist)
        grid_feats += mask * (feats - grid_feats) / denom
    return (grid_density, grid_feats) if features is not None else grid_density


@MODELS.register_module()
class GaussianVoxelizer(nn.Module):

    def __init__(self, vol_range, voxel_size):
        super().__init__()
        self.scene_shape = [
            int((vol_range[i + 3] - vol_range[i]) / voxel_size)
            for i in range(3)
        ]
        grid_coords = generate_grid(self.scene_shape, offset=0.5)
        grid_coords = grid_coords * voxel_size + torch.tensor(vol_range[:3])
        self.register_buffer('grid_coords', grid_coords)

    def forward(self, **gaussians):
        """
        gaussians:
            means3d, opacities, features, covariances / (scales, rotations)
        """
        # TODO: filter by opacities & out of boundaries
        # mask = ...
        # gaussians = apply_to_items(lambda x: x[mask], gaussians)

        if 'covariances' not in gaussians:
            gaussians['covariances'] = get_covariance(
                gaussians.pop('scales'),
                quat_to_rotmat(gaussians.pop('rotations')))
        outs = splat_into_3d(self.grid_coords, **gaussians)
        outs = [
            out.reshape(out.size(0), *self.scene_shape, out.size(-1))
            for out in (outs if isinstance(outs, tuple) else [outs])
        ]
        return outs if len(outs) > 1 else outs[0]
