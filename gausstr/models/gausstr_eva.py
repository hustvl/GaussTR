import numpy as np
import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmpretrain import get_model

from mmdet3d.registry import MODELS

from .utils import flatten_multi_scale_feats


@MODELS.register_module()
class GaussTR_EVA(BaseModel):

    def __init__(self, neck, decoder, num_queries, gauss_head, **kwargs):
        super().__init__(**kwargs)
        self.backbone = get_model(
            'vit-base-p14_eva02-pre_in21k',
            pretrained=True,
            backbone=dict(out_type='featmap'))
        self.neck = MODELS.build(neck)
        self.decoder = MODELS.build(decoder)

        self.query_embeds = nn.Embedding(
            num_queries, decoder.layer_cfg.self_attn_cfg.embed_dims)
        self.gauss_heads = nn.ModuleList(
            [MODELS.build(gauss_head) for _ in range(decoder.num_layers)])

        for param in self.backbone.parameters():
            param.requires_grad = False

    def prepare_inputs(self, inputs_dict, data_samples):
        num_views = data_samples[0].num_views
        imgs = inputs_dict['imgs']
        inputs_dict['imgs'] = imgs[:, :num_views]

        cam2img = []
        cam2ego = []
        ego2global = []
        img_aug_mat = []
        depth = []

        for i in range(len(data_samples)):
            data_samples[i].set_metainfo(
                {'cam2img': data_samples[i].cam2img[:num_views]})
            cam2img.append(data_samples[i].cam2img)
            data_samples[i].set_metainfo(
                {'cam2ego': data_samples[i].cam2ego[:num_views]})
            cam2ego.append(data_samples[i].cam2ego)
            ego2global.append(data_samples[i].ego2global)
            if hasattr(data_samples[i], 'img_aug_mat'):
                data_samples[i].set_metainfo(
                    {'img_aug_mat': data_samples[i].img_aug_mat[:num_views]})
                img_aug_mat.append(data_samples[i].img_aug_mat)
            depth.append(data_samples[i].depth)

        data_samples = dict(
            depth=depth,
            cam2img=cam2img,
            cam2ego=cam2ego,
            ego2global=ego2global,
            img_aug_mat=img_aug_mat if img_aug_mat else None)
        for k, v in data_samples.items():
            if k in ('imgs', ) or v is None:
                continue
            if isinstance(v[0], torch.Tensor):
                data_samples[k] = torch.stack(v).to(imgs)
            else:
                data_samples[k] = torch.from_numpy(np.stack(v)).to(imgs)
        return inputs_dict, data_samples

    def forward(self, inputs, data_samples, mode='loss'):
        if self.backbone.training:
            self.backbone.eval()
        inputs, data_samples = self.prepare_inputs(inputs, data_samples)
        with torch.no_grad():
            x = self.backbone.extract_feat(inputs['imgs'].flatten(0, 1))
        x = self.neck(x[0])

        decoder_inputs = self.pre_transformer(x)
        feats = flatten_multi_scale_feats(x)[0]
        decoder_inputs.update(self.pre_decoder(feats))
        decoder_outputs = self.forward_decoder(
            reg_branches=[h.reg_branch for h in self.gauss_heads],
            **decoder_inputs)

        query = decoder_outputs['hidden_states']
        reference_points = decoder_outputs['references']

        if mode == 'predict':
            return self.gauss_heads[-1](
                query[-1], reference_points[-1], mode=mode, **data_samples)

        losses = {}
        for i, gauss_head in enumerate(self.gauss_heads):
            loss = gauss_head(
                query[i], reference_points[i], mode=mode, **data_samples)
            for k, v in loss.items():
                losses[f'{k}/{i}'] = v
        return losses

    def pre_transformer(self, mlvl_feats):
        batch_size = mlvl_feats[0].size(0)

        mlvl_masks = []
        for feat in mlvl_feats:
            mlvl_masks.append(None)

        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask) in enumerate(zip(mlvl_feats, mlvl_masks)):
            batch_size, c, h, w = feat.shape
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            if mask is not None:
                mask = mask.flatten(1)

            feat_flatten.append(feat)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        if mask_flatten[0] is not None:
            mask_flatten = torch.cat(mask_flatten, 1)
        else:
            mask_flatten = None

        # (num_level, 2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        if mlvl_masks[0] is not None:
            valid_ratios = torch.stack(  # (bs, num_level, 2)
                [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        else:
            valid_ratios = mlvl_feats[0].new_ones(batch_size, len(mlvl_feats),
                                                  2)

        decoder_inputs_dict = dict(
            memory_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        return decoder_inputs_dict

    def pre_decoder(self, memory):
        bs, _, c = memory.shape
        query = self.query_embeds.weight.unsqueeze(0).expand(bs, -1, -1)
        reference_points = torch.rand((bs, query.size(1), 2)).to(query)

        decoder_inputs_dict = dict(
            query=query, memory=memory, reference_points=reference_points)
        return decoder_inputs_dict

    def forward_decoder(self, query, memory, memory_mask, reference_points,
                        spatial_shapes, level_start_index, valid_ratios,
                        **kwargs):
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)
        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict
