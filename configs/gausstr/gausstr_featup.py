_base_ = 'mmdet3d::_base_/default_runtime.py'

custom_imports = dict(imports=['gausstr'])

input_size = (432, 768)
embed_dims = 256

model = dict(
    type='GaussTR',
    num_queries=300,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]),
    neck=dict(
        type='ViTDetFPN',
        in_channels=512,
        out_channels=embed_dims,
        norm_cfg=dict(type='LN2d')),
    decoder=dict(
        type='GaussTRDecoder',
        num_layers=3,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=embed_dims, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=embed_dims, num_levels=4),
            ffn_cfg=dict(embed_dims=embed_dims, feedforward_channels=2048)),
        post_norm_cfg=None),
    gauss_head=dict(
        type='GaussTRCLIPHead',
        opacity_head=dict(
            type='MLP', input_dim=embed_dims, output_dim=1, mode='sigmoid'),
        feature_head=dict(type='MLP', input_dim=embed_dims, output_dim=512),
        scale_head=dict(
            type='MLP',
            input_dim=embed_dims,
            output_dim=3,
            mode='sigmoid',
            range=(1, 16)),
        regress_head=dict(type='MLP', input_dim=embed_dims, output_dim=3),
        text_protos='ckpts/text_proto_embeds.pth',
        reduce_dims=128,
        image_shape=input_size,
        voxelizer=dict(
            type='GaussianVoxelizer',
            vol_range=[-40, -40, -1, 40, 40, 5.4],
            voxel_size=0.4)))

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
    dict(type='LoadPrecompFeats', root_dir='data/nuscenes_featup'),
    dict(
        type='Pack3DDetInputs',
        keys=['img'],
        meta_keys=[
            'cam2img', 'cam2ego', 'ego2global', 'img_aug_mat', 'sample_idx',
            'num_views', 'img_path', 'depth', 'feats'
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
    dict(type='LoadPrecompFeats', root_dir='data/nuscenes_featup'),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=[
            'cam2img', 'cam2ego', 'ego2global', 'img_aug_mat', 'sample_idx',
            'num_views', 'img_path', 'depth', 'feats', 'mask_camera'
        ])
]

shared_dataset_cfg = dict(
    type=dataset_type,
    data_root=data_root,
    modality=input_modality,
    data_prefix=data_prefix,
    filter_empty_gt=False)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        ann_file='nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        **shared_dataset_cfg))
val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
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
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=5e-3),
    clip_grad=dict(max_norm=35, norm_type=2))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=200, by_epoch=False),
    dict(type='MultiStepLR', milestones=[16], gamma=0.1)
]
