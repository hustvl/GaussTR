import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.models import inverse_sigmoid
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS

from .utils import cam2world, get_covariance, rotmat_to_quat


def flatten_bsn_forward(func, *args, **kwargs):
    args = list(args)
    bsn = None
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            if bsn is None:
                bsn = arg.shape[:2]
            args[i] = arg.flatten(0, 1)
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            if bsn is None:
                bsn = v.shape[:2]
            kwargs[k] = v.flatten(0, 1)
    outs = func(*args, **kwargs)
    if isinstance(outs, tuple):
        for i, out in outs:
            outs[i] = out.reshape(bsn + out.shape[1:])
    else:
        outs = outs.reshape(bsn + outs.shape[1:])
    return outs


@MODELS.register_module()
class GaussTRCLIPHead(BaseModule):

    def __init__(self,
                 opacity_head,
                 feature_head,
                 scale_head,
                 regress_head,
                 image_shape,
                 rasterizer,
                 voxelizer,
                 depth_limit=51.2,
                 visual_projection=None,
                 text_protos=None):
        super().__init__()
        self.opacity_head = MODELS.build(opacity_head)
        self.feature_head = MODELS.build(feature_head)
        self.scale_head = MODELS.build(scale_head)
        self.regress_head = MODELS.build(regress_head)
        self.image_shape = image_shape
        self.depth_limit = depth_limit

        if visual_projection is not None:
            self.visual_projection = MODELS.build(visual_projection)
            if 'init_cfg' in visual_projection and visual_projection.init_cfg.type == 'Pretrained':
                self.visual_projection.requires_grad_(False)
        if text_protos is not None:
            self.register_buffer('text_proto_embeds',
                                 torch.load(text_protos, map_location='cpu'))

        self.rasterizer = MODELS.build(rasterizer)
        self.voxelizer = MODELS.build(voxelizer)
        self.silog_loss = MODELS.build(dict(type='SiLogLoss', _scope_='mmseg'))

    def forward(self,
                x,
                ref_pts,
                depth,
                cam2img,
                cam2ego,
                mode='tensor',
                imgs=None,
                **kwargs):
        bs, n = cam2img.shape[:2]
        x = x.reshape(bs, n, *x.shape[1:])
        ref_pts = (
            self.regress_head(x) +
            inverse_sigmoid(ref_pts.reshape(*x.shape[:-1], -1))).sigmoid()
        depth = depth.clamp(max=self.depth_limit)
        sample_depth = flatten_bsn_forward(F.grid_sample, depth[:, :n, None],
                                           ref_pts.unsqueeze(2) * 2 - 1)
        sample_depth = sample_depth[:, :, 0, 0, :, None]
        points = torch.cat([
            ref_pts * torch.tensor(self.image_shape[::-1]).to(x), sample_depth
        ], -1)

        means3d = cam2world(points, cam2img, cam2ego, kwargs['img_aug_mat'])
        opacities = self.opacity_head(x)
        scales = self.scale_head(x) * self.scale_transform(
            sample_depth, cam2img[..., 0, 0]).clamp(1e-6)
        features = self.feature_head(x).float()

        covariances = flatten_bsn_forward(get_covariance, scales,
                                          cam2ego[..., None, :3, :3])
        rotations = flatten_bsn_forward(rotmat_to_quat, cam2ego[..., :3, :3])
        rotations = rotations.unsqueeze(2).expand(-1, -1, x.size(2), -1)

        if mode == 'predict':
            features = features @ self.text_proto_embeds

            density, grid_feats = self.voxelizer(
                means3d=means3d.flatten(1, 2),
                opacities=opacities.flatten(1, 2),
                features=features.flatten(1, 2).softmax(-1),
                covariances=covariances.flatten(1, 2))
            pred = torch.where(pred.squeeze(-1) > 0.0625, 1, 17)  # 4e-2 ~ 1/8
            return pred

        tgt_feats = imgs.flatten(2).mT
        tgt_feats = self.visual_projection(tgt_feats)[0]
        u, s, v = torch.pca_lowrank(
            tgt_feats.flatten(0, 1).double(), q=32, niter=4)
        tgt_feats = tgt_feats @ v.to(tgt_feats)
        features = features @ v.to(features)
        features = features.float()

        rendered_imgs, rendered_depth = self.rasterizer(
            means3d.flatten(1, 2),
            features.flatten(1, 2),
            opacities.flatten(1, 2),
            scales=scales.flatten(1, 2),
            rotations=rotations.flatten(1, 2),
            img_shape=(900, 1600),  # TODO
            cam2img=cam2img,
            cam2ego=cam2ego,
            img_aug_mat=kwargs['img_aug_mat'])
        # self.visualize_rendered_results((rendered_depth, depth.unsqueeze(2)))

        rendered_imgs = rendered_imgs[:, :6].flatten(0, 1)
        rendered_imgs = F.avg_pool2d(rendered_imgs, 16)
        rendered_imgs = rendered_imgs.flatten(2).mT

        depth = torch.where(depth < self.depth_limit, depth, 1e-3)
        losses = {}
        losses['loss_depth'] = self.depth_loss(
            rendered_depth.flatten(0, 2), depth[:, :6].flatten(0, 1))
        losses['mae_depth'] = self.depth_loss(
            rendered_depth.flatten(0, 2),
            depth[:, :6].flatten(0, 1),
            criterion='l1')
        losses['loss_cosine'] = F.cosine_embedding_loss(
            rendered_imgs.flatten(0, 1), tgt_feats.flatten(0, 1),
            torch.ones_like(tgt_feats.flatten(0, 1)[:, 0])) * 2
        losses['loss_mse'] = F.mse_loss(rendered_imgs, tgt_feats) * 2
        return losses

    def photometric_error(self, src_imgs, rec_imgs):
        return (0.85 * self.ssim(src_imgs, rec_imgs) +
                0.15 * F.l1_loss(src_imgs, rec_imgs))

    def depth_loss(self, pred, target, criterion='silog_l1'):
        loss = 0
        if 'silog' in criterion:
            loss += self.silog_loss(pred, target)
        if 'l1' in criterion:
            target = target.flatten()
            pred = pred.flatten()[target != 0]
            l1_loss = F.l1_loss(pred, target[target != 0])
            if loss != 0:
                l1_loss *= 0.2
            loss += l1_loss
        return loss

    def scale_transform(self, depth, focal, multiplier=7.5):
        return depth * multiplier / focal.reshape(*depth.shape[:2], 1, 1)

    def compute_ref_params(self, cam2img, cam2ego, ego2global, img_aug_mat):
        ego2keyego = torch.inverse(ego2global[:, 0:1]) @ ego2global[:, 1:]
        cam2keyego = ego2keyego.unsqueeze(2) @ cam2ego.unsqueeze(1)
        cam2keyego = torch.cat([cam2ego.unsqueeze(1), cam2keyego],
                               dim=1).flatten(1, 2)
        cam2img = cam2img.unsqueeze(1).expand(-1, 3, -1, -1, -1).flatten(1, 2)
        img_aug_mat = img_aug_mat.unsqueeze(1).expand(-1, 3, -1, -1,
                                                      -1).flatten(1, 2)
        return dict(
            cam2img=cam2img, cam2ego=cam2keyego, img_aug_mat=img_aug_mat)

    def visualize_rendered_results(self, results):
        # (bs, t*n, 3/1, h, w)
        vis = []
        for res in results:
            res = res[0]
            if res.dim() == 3:
                res = res.reshape(
                    res.size(0), 1, -1, vis[0].size(1) // self.downsample)
                res = res.unsqueeze(0).expand(3, *([-1] * 4)).flatten(0, 1)
                res = F.interpolate(res, scale_factor=self.downsample)

            img = res.permute(0, 2, 3, 1).flatten(
                0, 1).detach().cpu().numpy()  # (t * n * h, w, 3/1)
            if img.shape[-1] == 1:
                from matplotlib import colormaps as cm
                cmap = cm.get_cmap('Spectral_r')
                img = cmap(img / (img.max() + 1e-5))[..., 0, :3]
            img -= img.min()
            img /= img.max()
            vis.append(img)

        vis = np.concatenate(vis, axis=-2)
        plt.imsave(f'{time.time() % 1e6:.2f}.png', vis)
