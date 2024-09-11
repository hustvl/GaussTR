import numpy as np
import torch
import torch.nn as nn
from mmengine.model import BaseModel, ModuleList

from mmdet3d.registry import MODELS


@MODELS.register_module()
class GaussTR(BaseModel):

    def __init__(self, backbone, neck, encoder, decoder, num_queries,
                 gauss_head, positional_encoding, **kwargs):
        super().__init__(**kwargs)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.encoder = MODELS.build(encoder)
        self.decoder = MODELS.build(decoder)
        self.positional_encoding = MODELS.build(positional_encoding)
        self.level_embed = nn.Parameter(
            torch.Tensor(encoder.layer_cfg.self_attn_cfg.num_levels,
                         encoder.layer_cfg.self_attn_cfg.embed_dims))
        self.query_embeds = nn.Embedding(
            num_queries, decoder.layer_cfg.self_attn_cfg.embed_dims)
        self.gauss_heads = ModuleList(
            [MODELS.build(gauss_head) for _ in range(decoder.num_layers)])

        self.frozen_backbone = all(not param.requires_grad
                                   for param in self.backbone.parameters())
        self.return_values = backbone.out_indices == -2

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
        inputs, data_samples = self.prepare_inputs(inputs, data_samples)
        if self.frozen_backbone:
            if self.backbone.training:
                self.backbone.eval()
            with torch.no_grad():
                x = self.backbone(inputs['imgs'].flatten(0, 1))[0]
                if self.return_values:
                    x = self.forward_values(x)
        else:
            x = self.backbone(inputs['imgs'].flatten(0, 1))[0]
        feats = self.neck(x)

        encoder_inputs, decoder_inputs = self.pre_transformer(feats)
        feats = self.forward_encoder(**encoder_inputs)
        decoder_inputs.update(self.pre_decoder(feats))
        decoder_outputs = self.forward_decoder(
            reg_branches=[h.regress_head for h in self.gauss_heads],
            **decoder_inputs)

        query = decoder_outputs['hidden_states']
        reference_points = decoder_outputs['references']

        if mode == 'predict':
            return self.gauss_heads[-1](
                query[-1], reference_points[-1], mode=mode, **data_samples)

        losses = {}
        for i, gauss_head in enumerate(self.gauss_heads):
            loss = gauss_head(
                query[i],
                reference_points[i],
                mode=mode,
                imgs=x,
                **data_samples)
            for k, v in loss.items():
                losses[f'{k}/{i}'] = v
        return losses

    def forward_values(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).mT
        last_layer = self.backbone.layers[-1]
        qkv = last_layer.attn.qkv(last_layer.ln1(x)).reshape(
            B, H * W, 3, last_layer.attn.embed_dims)
        v = last_layer.attn.proj(qkv[:, :, 2])
        v += x
        v = last_layer.ffn(last_layer.ln2(v), identity=v)

        if self.backbone.final_norm:
            v = self.backbone.ln1(v)
        return v.reshape(B, H, W, C).permute(0, 3, 1, 2)

    def pre_transformer(self, mlvl_feats):
        batch_size = mlvl_feats[0].size(0)

        mlvl_masks = []
        mlvl_pos_embeds = []
        for feat in mlvl_feats:
            mlvl_masks.append(None)
            mlvl_pos_embeds.append(self.positional_encoding(None, input=feat))

        feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            batch_size, c, h, w = feat.shape
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            if mask is not None:
                mask = mask.flatten(1)

            feat_flatten.append(feat)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
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

        encoder_inputs_dict = dict(
            feat=feat_flatten,
            feat_mask=mask_flatten,
            feat_pos=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        decoder_inputs_dict = dict(
            memory_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, feat, feat_mask, feat_pos, spatial_shapes,
                        level_start_index, valid_ratios):
        memory = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        return memory

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
