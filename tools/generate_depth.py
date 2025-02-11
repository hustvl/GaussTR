import os.path as osp
import numpy as np
import torch

from mmengine import FUNCTIONS, Config
from mmdet3d.registry import DATASETS, MODELS
try:
    from torchvision.transforms import v2 as T
except:
    from torchvision import transforms as T
from torch.utils.data import DataLoader
from rich.progress import track

from gausstr import *


def test_loop(model, dataset_cfg, dataloader_cfg, save_dir):
    dataset = DATASETS.build(dataset_cfg)
    dataloader = DataLoader(
        dataset, collate_fn=FUNCTIONS.get('pseudo_collate'), **dataloader_cfg)
    transform = T.Normalize(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

    for data in track(dataloader):
        data_samples = data['data_samples']
        cam2imgs = []
        img_aug_mats = []
        img_paths = []
        for i in range(len(data_samples)):
            data_samples[i].set_metainfo({'cam2img': data_samples[i].cam2img})
            cam2imgs.append(data_samples[i].cam2img)
            if hasattr(data_samples[i], 'img_aug_mat'):
                data_samples[i].set_metainfo(
                    {'img_aug_mat': data_samples[i].img_aug_mat})
                img_aug_mats.append(data_samples[i].img_aug_mat)
            img_paths.append(data_samples[i].img_path)
        cam2imgs = torch.from_numpy(np.concatenate(cam2imgs)).cuda()
        if img_aug_mats:
            img_aug_mats = torch.from_numpy(np.concatenate(img_aug_mats)).cuda()
        img_paths = sum(img_paths, [])
        x = transform(torch.cat(data['inputs']['img']).cuda())

        with torch.no_grad():
            depths = model(x, cam2imgs)
        depths = depths.cpu().numpy()
        for path, depth in zip(img_paths, depths):
            save_path = osp.join(save_dir, path.split('/')[-1].split('.')[0])
            np.save(save_path, depth)


if __name__ == '__main__':
    ann_files = [
        'nuscenes_infos_train.pkl', 'nuscenes_infos_val.pkl',
        # 'nuscenes_infos_mini_train.pkl', 'nuscenes_infos_mini_val.pkl'
    ]
    cfg = Config.fromfile('configs/gausstr_featup.py')
    model = MODELS.build(
        dict(type='Metric3D', model_name='metric3d_vit_large')).cuda()
    save_dir = 'data/nuscenes_metric3d'

    dataloader_cfg = cfg.test_dataloader
    dataloader_cfg.pop('sampler')
    dataset_cfg = dataloader_cfg.pop('dataset')
    dataset_cfg.pipeline = [
        t | dict(_scope_='mmdet3d') for t in dataset_cfg.pipeline
        if t.type in ('BEVLoadMultiViewImageFromFiles', 'Pack3DDetInputs')  # 'ImageAug3D'
    ]

    for ann_file in ann_files:
        dataset_cfg.ann_file = ann_file
        test_loop(model, dataset_cfg, cfg.test_dataloader, save_dir)
