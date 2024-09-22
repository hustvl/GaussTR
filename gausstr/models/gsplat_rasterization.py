import torch
from gsplat import rasterization

from .utils import unbatched_forward


@unbatched_forward
def rasterize_gaussians(means3d,
                        colors,
                        opacities,
                        scales,
                        rotations,
                        cam2imgs,
                        cam2egos,
                        image_size,
                        img_aug_mats=None,
                        **kwargs):
    # cam2world to world2cam
    R = cam2egos[:, :3, :3].mT
    T = -R @ cam2egos[:, :3, 3:4]
    viewmat = torch.zeros_like(cam2egos)
    viewmat[:, :3, :3] = R
    viewmat[:, :3, 3:] = T
    viewmat[:, 3, 3] = 1

    if img_aug_mats is not None:
        cam2imgs = cam2imgs.clone()
        cam2imgs[:, :2, :2] *= img_aug_mats[:, :2, :2]
        image_size = list(image_size)
        for i in range(2):
            cam2imgs[:, i, 2] *= img_aug_mats[:, i, i]
            cam2imgs[:, i, 2] += img_aug_mats[:, i, 3]
            image_size[1 - i] = round(image_size[1 - i] *
                                      img_aug_mats[0, i, i].item() +
                                      img_aug_mats[0, i, 3].item())

    rendered_image = rasterization(
        means3d,
        rotations,
        scales,
        opacities,
        colors,
        viewmat,
        cam2imgs,
        width=image_size[1],
        height=image_size[0],
        **kwargs)[0]
    return rendered_image.permute(0, 3, 1, 2)
