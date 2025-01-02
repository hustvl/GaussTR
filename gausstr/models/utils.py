import numpy as np
import torch
from pyquaternion import Quaternion
from torch.cuda.amp import autocast


def nlc_to_nchw(x, shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.
    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        shape (Sequence[int]): The height and width of output feature map.
    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    B, L, C = x.shape
    return x.transpose(1, 2).reshape(B, C, *shape).contiguous()


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.
    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.
    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
        tuple: The [H, W] shape.
    """
    return x.flatten(2).transpose(1, 2).contiguous()


def flatten_multi_scale_feats(feats):
    feat_flatten = torch.cat([nchw_to_nlc(feat) for feat in feats], dim=1)
    shapes = torch.stack([
        torch.tensor(feat.shape[2:], device=feat_flatten.device)
        for feat in feats
    ])
    return feat_flatten, shapes


def get_level_start_index(shapes):
    return torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))


def generate_grid(grid_shape, value=None, offset=0, normalize=False):
    """
    Args:
        grid_shape: The (scaled) shape of grid.
        value: The (unscaled) value the grid represents.
    Returns:
        Grid coordinates of shape [*grid_shape, len(grid_shape)]
    """
    if value is None:
        value = grid_shape
    grid = []
    for i, (s, val) in enumerate(zip(grid_shape, value)):
        g = torch.linspace(offset, val - 1 + offset, s, dtype=torch.float)
        if normalize:
            g /= val
        shape_ = [1 for _ in grid_shape]
        shape_[i] = s
        g = g.reshape(*shape_).expand(*grid_shape)
        grid.append(g)
    return torch.stack(grid, dim=-1)


def cam2world(points, cam2img, cam2ego, img_aug_mat=None):
    if img_aug_mat is not None:
        post_rots = img_aug_mat[..., :3, :3]
        post_trans = img_aug_mat[..., :3, 3]
        points = points - post_trans.unsqueeze(-2)
        points = (torch.inverse(post_rots).unsqueeze(2)
                  @ points.unsqueeze(-1)).squeeze(-1)

    cam2img = cam2img[..., :3, :3]
    with autocast(enabled=False):
        combine = cam2ego[..., :3, :3] @ torch.inverse(cam2img)
        points = points.float()
        points = torch.cat(
            [points[..., :2] * points[..., 2:3], points[..., 2:3]], dim=-1)
        points = combine.unsqueeze(2) @ points.unsqueeze(-1)
    points = points.squeeze(-1) + cam2ego[..., None, :3, 3]
    return points


def world2cam(points, cam2img, cam2ego, img_aug_mat=None, eps=1e-6):
    points = points - cam2ego[..., None, :3, 3]
    points = torch.inverse(cam2ego[..., None, :3, :3]) @ points.unsqueeze(-1)
    points = (cam2img[..., None, :3, :3] @ points).squeeze(-1)
    points = points / points[..., 2:3].clamp(eps)  # NOTE
    if img_aug_mat is not None:
        points = img_aug_mat[..., None, :3, :3] @ points.unsqueeze(-1)
        points = points.squeeze(-1) + img_aug_mat[..., None, :3, 3]
    return points[..., :2]


def rotmat_to_quat(rot_matrices):
    inputs = rot_matrices
    rot_matrices = rot_matrices.cpu().numpy()
    quats = []
    for rot in rot_matrices:
        while not np.allclose(rot @ rot.T, np.eye(3)):
            U, _, V = np.linalg.svd(rot)
            rot = U @ V
        quats.append(Quaternion(matrix=rot).elements)
    return torch.from_numpy(np.stack(quats)).to(inputs)


def quat_to_rotmat(quats):
    q = quats / torch.sqrt((quats**2).sum(dim=-1, keepdim=True))
    r, x, y, z = [i.squeeze(-1) for i in q.split(1, dim=-1)]

    R = torch.zeros((*r.shape, 3, 3)).to(r)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - r * z)
    R[..., 0, 2] = 2 * (x * z + r * y)
    R[..., 1, 0] = 2 * (x * y + r * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - r * x)
    R[..., 2, 0] = 2 * (x * z - r * y)
    R[..., 2, 1] = 2 * (y * z + r * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def get_covariance(s, r):
    L = torch.zeros((*s.shape[:2], 3, 3)).to(s)
    for i in range(s.size(-1)):
        L[..., i, i] = s[..., i]

    L = r @ L
    covariance = L @ L.mT
    return covariance


def unbatched_forward(func):

    def wrapper(*args, **kwargs):
        bs = None
        for arg in list(args) + list(kwargs.values()):
            if isinstance(arg, torch.Tensor):
                if bs is None:
                    bs = arg.size(0)
                else:
                    assert bs == arg.size(0)

        outputs = []
        for i in range(bs):
            output = func(
                *[
                    arg[i] if isinstance(arg, torch.Tensor) else arg
                    for arg in args
                ], **{
                    k: v[i] if isinstance(v, torch.Tensor) else v
                    for k, v in kwargs.items()
                })
            outputs.append(output)

        if isinstance(outputs[0], tuple):
            return tuple([
                torch.stack([out[i] for out in outputs])
                for i in range(len(outputs[0]))
            ])
        else:
            return torch.stack(outputs)

    return wrapper


OCC3D_CATEGORIES = (
    ['barrier'],
    ['bicycle'],
    ['bus'],
    ['car'],
    ['construction vehicle'],
    ['motorcycle'],
    ['person'],
    ['cone'],
    ['trailer'],
    ['truck'],
    ['road'],
    ['sidewalk'],
    ['terrain', 'grass'],
    ['building', 'wall', 'fence', 'pole', 'sign'],
    ['vegetation'],
    ['sky'],
)  # `sum(OCC3D_CATEGORIES, [])` if you need to flatten the list.
