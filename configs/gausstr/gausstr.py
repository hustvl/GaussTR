_base_ = 'mmdet3d::_base_/default_runtime.py'

custom_imports = dict(imports=['gausstr'])

input_size = (396, 768)

model = dict(
    type='GaussTR',
    num_queries=300,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]),
    backbone=dict(
        type='ResNet',
        _scope_='mmdet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        _scope_='mmdet',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
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
        type='GaussTRHead',
        opacity_head=dict(type='Scaler', input_dim=256),
        scale_head=dict(
            type='Scaler', input_dim=256, output_dim=3, range=(1, 16)),
        image_shape=input_size,
        rasterizer=dict(type='GaussianRasterizer'),
        voxelizer=dict(
            type='GaussianVoxelizer',
            vol_range=[-40, -40, -1, 40, 40, 5.4],
            voxel_size=0.4)),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=
        'ckpts/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth'
    ))

# Data
dataset_type = 'NuScenesOccDataset'
data_root = 'data/nuscenes/'
data_prefix = dict(
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT')
input_modality = dict(use_camera=True, use_lidar=False)

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

shared_dataset_cfg = dict(
    type=dataset_type,
    data_root=data_root,
    modality=input_modality,
    data_prefix=data_prefix,
    filter_empty_gt=False)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        ann_file='nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        # load_adj_frame=True,
        **shared_dataset_cfg))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        ann_file='nuscenes_infos_mini_val.pkl',  # TODO
        pipeline=test_pipeline,
        # load_adj_frame=True,
        **shared_dataset_cfg))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='OccMetric',
    num_classes=18,
    use_lidar_mask=False,
    use_image_mask=True)
test_evaluator = val_evaluator

# Optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=5e-3),
    clip_grad=dict(max_norm=35, norm_type=2))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=200, by_epoch=False),
    dict(type='ConstantLR', factor=1)  # TODO: MultiStepLR
]

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))  # TODO
