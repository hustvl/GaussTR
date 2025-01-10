import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import inverse_sigmoid
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS

from .gsplat_rasterization import rasterize_gaussians
from .utils import OCC3D_CATEGORIES, cam2world, get_covariance, rotmat_to_quat


@MODELS.register_module()
class MLP(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dim=None,
                 output_dim=None,
                 num_layers=2,
                 activation='relu',
                 mode=None,
                 range=None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim * 4
        output_dim = output_dim or input_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.activation = activation
        self.range = range
        self.mode = mode

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = getattr(F, self.activation)(
                layer(x)) if i < self.num_layers - 1 else layer(x)

        if self.mode is not None:
            if self.mode == 'sigmoid':
                x = F.sigmoid(x)
            if self.range is not None:
                x = self.range[0] + (self.range[1] - self.range[0]) * x
        return x


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
        outs = list(outs)
        for i, out in outs:
            outs[i] = out.reshape(bsn + out.shape[1:])
    else:
        outs = outs.reshape(bsn + outs.shape[1:])
    return outs


def prompt_denoising(logits, logit_scale=100, pd_threshold=0.1):
    probs = logits.softmax(-1)
    probs_ = F.softmax(logits * logit_scale, -1)
    max_cls_conf = probs_.flatten(1, 3).max(1)[0]
    selected_cls = (max_cls_conf < pd_threshold)[:, None, None,
                                                 None].expand(*probs.shape)
    probs[selected_cls] = 0
    return probs


def merge_probs(probs, categories):
    merged_probs = []
    i = 0
    for cats in categories:
        p = probs[..., i:i + len(cats)]
        i += len(cats)
        if len(cats) > 1:
            p = p.max(-1, keepdim=True)[0]
        merged_probs.append(p)
    return torch.cat(merged_probs, dim=-1)


@MODELS.register_module()
class GaussTRHead(BaseModule):

    def __init__(self,
                 opacity_head,
                 feature_head,
                 scale_head,
                 regress_head,
                 reduce_dims,
                 image_shape,
                 voxelizer,
                 segment_head=None,
                 depth_limit=51.2,
                 projection=None,
                 text_protos=None,
                 prompt_denoising=True):
        super().__init__()
        self.opacity_head = MODELS.build(opacity_head)
        self.feature_head = MODELS.build(feature_head)
        self.scale_head = MODELS.build(scale_head)
        self.regress_head = MODELS.build(regress_head)
        self.segment_head = MODELS.build(
            segment_head) if segment_head else None

        self.reduce_dims = reduce_dims
        self.image_shape = image_shape
        self.depth_limit = depth_limit
        self.prompt_denoising = prompt_denoising

        if projection is not None:
            self.projection = MODELS.build(projection)
            if 'init_cfg' in projection and projection.init_cfg.type == 'Pretrained':
                self.projection.requires_grad_(False)
        if text_protos is not None:
            self.register_buffer('text_proto_embeds',
                                 torch.load(text_protos, map_location='cpu'))

        self.voxelizer = MODELS.build(voxelizer)
        self.silog_loss = MODELS.build(dict(type='SiLogLoss', _scope_='mmseg'))

    def forward(self,
                x,
                ref_pts,
                depth,
                cam2img,
                cam2ego,
                mode='tensor',
                feats=None,
                img_aug_mat=None,
                sem_segs=None,
                **kwargs):
        bs, n = cam2img.shape[:2]
        x = x.reshape(bs, n, *x.shape[1:])

        deltas = self.regress_head(x)
        ref_pts = (
            deltas[..., :2] +
            inverse_sigmoid(ref_pts.reshape(*x.shape[:-1], -1))).sigmoid()
        depth = depth.clamp(max=self.depth_limit)
        sample_depth = flatten_bsn_forward(F.grid_sample, depth[:, :n, None],
                                           ref_pts.unsqueeze(2) * 2 - 1)
        sample_depth = sample_depth[:, :, 0, 0, :, None]
        points = torch.cat([
            ref_pts * torch.tensor(self.image_shape[::-1]).to(x),
            sample_depth * (1 + deltas[..., 2:3])
        ], -1)
        means3d = cam2world(points, cam2img, cam2ego, img_aug_mat)

        opacities = self.opacity_head(x).float()
        features = self.feature_head(x).float()
        scales = self.scale_head(x) * self.scale_transform(
            sample_depth, cam2img[..., 0, 0]).clamp(1e-6)

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
            if self.prompt_denoising:
                probs = prompt_denoising(grid_feats)
            else:
                probs = grid_feats.softmax(-1)

            probs = merge_probs(probs, OCC3D_CATEGORIES)
            preds = probs.argmax(-1)
            preds += (preds > 10) * 1 + 1  # skip two classes of "others"
            preds = torch.where(density.squeeze(-1) > 4e-2, preds, 17)
            return preds

        tgt_feats = feats.flatten(-2).mT
        if hasattr(self, 'projection'):
            tgt_feats = self.projection(tgt_feats)[0]

        u, s, v = torch.pca_lowrank(
            tgt_feats.double(), q=self.reduce_dims, niter=4)
        tgt_feats = tgt_feats @ v.to(tgt_feats)
        features = features @ v.to(features)
        features = features.float()

        rendered = rasterize_gaussians(
            means3d.flatten(1, 2),
            features.flatten(1, 2),
            opacities.squeeze(-1).flatten(1, 2),
            scales.flatten(1, 2),
            rotations.flatten(1, 2),
            cam2img,
            cam2ego,
            img_aug_mats=img_aug_mat,
            image_size=(900, 1600),
            near_plane=0.1,
            far_plane=100,
            render_mode='RGB+D',  # NOTE: 'ED' mode is better for visualization
            channel_chunk=32).flatten(0, 1)
        rendered_depth = rendered[:, -1]
        rendered = rendered[:, :-1]

        losses = {}
        depth = torch.where(depth < self.depth_limit, depth,
                            1e-3).flatten(0, 1)
        losses['loss_depth'] = self.depth_loss(rendered_depth, depth)
        losses['mae_depth'] = self.depth_loss(
            rendered_depth, depth, criterion='l1')

        rendered = rendered.flatten(2).mT
        tgt_feats = tgt_feats.mT.reshape(bs * n, self.reduce_dims,
                                         *feats.shape[-2:])
        tgt_feats = F.interpolate(tgt_feats, scale_factor=16, mode='bilinear')
        tgt_feats = tgt_feats.flatten(2).mT
        losses['loss_cosine'] = F.cosine_embedding_loss(
            rendered.flatten(0, 1), tgt_feats.flatten(0, 1),
            torch.ones_like(tgt_feats.flatten(0, 1)[:, 0])) * 5

        if self.segment_head:
            losses['loss_ce'] = F.cross_entropy(
                self.segment_head(rendered).mT,
                sem_segs.flatten(0, 1).flatten(1).long(),
                ignore_index=0)
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
            cam2imgs=cam2img, cam2egos=cam2keyego, img_aug_mats=img_aug_mat)

    def visualize_rendered_results(self,
                                   results,
                                   arrangement='vertical',
                                   save_dir='vis'):
        # (bs, t*n, 3/1, h, w)
        assert arrangement in ('vertical', 'tiled')
        if not isinstance(results, (list, tuple)):
            results = [results]
        vis = []
        for res in results:
            res = res[0]
            if res.dim() == 3:
                res = res.reshape(
                    res.size(0), 1, -1, vis[0].size(1) // self.downsample)
                res = res.unsqueeze(0).expand(3, *([-1] * 4)).flatten(0, 1)
                res = F.interpolate(res, scale_factor=self.downsample)

            img = res.permute(0, 2, 3, 1)  # (t * n, h, w, 3/1)
            if arrangement == 'vertical':
                img = img.flatten(0, 1)
            else:
                img = torch.cat((
                    torch.cat((img[2], img[4]), dim=0),
                    torch.cat((img[0], img[3]), dim=0),
                    torch.cat((img[1], img[5]), dim=0),
                ),
                                dim=1)
            img = img.detach().cpu().numpy()
            if img.shape[-1] == 1:
                from matplotlib import colormaps as cm
                cmap = cm.get_cmap('Spectral_r')
                img = cmap(img / (img.max() + 1e-5))[..., 0, :3]
            img -= img.min()
            img /= img.max()
            vis.append(img)
        vis = np.concatenate(vis, axis=-2)

        if not hasattr(self, 'save_cnt'):
            self.save_cnt = 0
        else:
            self.save_cnt += 1
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        plt.imsave(osp.join(save_dir, f'{self.save_cnt}.png'), vis)
