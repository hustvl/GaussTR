_base_ = './gausstr.py'

custom_imports = dict(imports=['gausstr'])

input_size = (432, 768)

model = dict(
    type='GaussTR',
    _delete_=True,
    num_queries=300,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]),
    backbone=dict(
        type='VisionTransformer',
        _scope_='mmpretrain',
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.,
        out_type='featmap',
        layer_cfgs=dict(act_cfg=dict(type='QuickGELU')),
        pre_norm=True,
        frozen_stages=12,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='ckpts/clip-vit-b-p16.pth',
            prefix='visual')),
    neck=dict(
        type='ViTDetFPN',
        in_channels=768,
        out_channels=256,
        norm_cfg=dict(type='LN2d')),
    encoder=dict(
        type='DeformableDetrTransformerEncoder',
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048))),
    decoder=dict(
        type='GaussTRDecoder',
        num_layers=3,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels=4),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048)),
        post_norm_cfg=None),
    positional_encoding=dict(
        type='SinePositionalEncoding',
        _scope_='mmdet',
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20),
    gauss_head=dict(
        type='GaussTRCLIPHead',
        opacity_head=dict(type='Scaler', input_dim=256),
        scale_head=dict(
            type='Scaler', input_dim=256, output_dim=3, range=(1, 16)),
        visual_projection=dict(
            type='CLIPProjection',
            _scope_='mmpretrain',
            in_channels=768,
            out_channels=512,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='ckpts/clip-vit-b-p16.pth',
                prefix='visual_proj')),
        text_protos='ckpts/text_proto_embeds.pth',
        image_shape=input_size,
        rasterizer=dict(type='GaussianRasterizer'),
        voxelizer=dict(
            type='GaussianVoxelizer',
            vol_range=[-40, -40, -1, 40, 40, 5.4],
            voxel_size=0.4)))

# Data
train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        num_views=6),
    dict(
        type='ImageAug3D',
        final_dim=input_size,
        resize_lim=[0.48, 0.48],
        is_train=True),
    dict(type='LoadDepthPreds', depth_root='data/nuscenes_depth_metric3d'),
    dict(
        type='Pack3DDetInputs',
        keys=['img'],
        meta_keys=[
            'cam2img', 'cam2ego', 'ego2global', 'img_aug_mat', 'sample_idx',
            'num_views', 'img_path', 'depth', 'mask_camera'
        ])
]
test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        num_views=6),
    dict(type='LoadOccGTFromFile'),
    dict(type='ImageAug3D', final_dim=input_size, resize_lim=[0.48, 0.48]),
    dict(type='LoadDepthPreds', depth_root='data/nuscenes_depth_metric3d'),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=[
            'cam2img', 'cam2ego', 'ego2global', 'img_aug_mat', 'sample_idx',
            'num_views', 'img_path', 'depth', 'mask_camera'
        ])
]

train_dataloader = dict(batch_size=4, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=4, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
