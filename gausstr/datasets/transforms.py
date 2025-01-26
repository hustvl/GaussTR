import copy
import os
from typing import Optional

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.transforms import BaseTransform
from mmengine.fileio import get
from PIL import Image

from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class BEVLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.

    ``BEVLoadMultiViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # Support multi-view images with different shapes
        filename, cam2img, lidar2cam, cam2ego = [], [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            lidar2cam.append(cam_item['lidar2cam'])

            cam2img_array = np.eye(4).astype(np.float32)
            cam2img_array[:3, :3] = np.array(cam_item['cam2img']).astype(
                np.float32)
            cam2img.append(cam2img_array)

            cam2ego_array = np.array(cam_item['cam2ego']).astype(np.float32)
            cam2ego.append(cam2ego_array)

        results['img_path'] = filename
        results['cam2img'] = np.stack(cam2img, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0)
        results['cam2ego'] = np.stack(cam2ego, axis=0)

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        imgs = [
            mmcv.imfrombytes(
                img_byte,
                flag=self.color_type,
                backend='pillow',
                channel_order='rgb') for img_byte in img_bytes
        ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        return results


@TRANSFORMS.register_module()
class PointToMultiViewDepth(BaseTransform):

    def __init__(self, depth_cfg, downsample=1):
        self.downsample = downsample
        self.depth_cfg = depth_cfg

    def points2depth(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width))
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]

        kept1 = ((coor[:, 0] >= 0) & (coor[:, 0] < width) & (coor[:, 1] >= 0) &
                 (coor[:, 1] < height) & (depth < self.depth_cfg[1]) &
                 (depth >= self.depth_cfg[0]))
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth.to(depth_map)
        return depth_map

    def transform(self, results):
        pts_lidar = results['points']
        imgs = results['img']
        cam2imgs = results['cam2img']
        img_aug_mats = results['img_aug_mat']
        depth = []

        for i, cam_name in enumerate(results['images']):
            cam2img = cam2imgs[i]
            lidar2cam = results['images'][cam_name]['lidar2cam']
            lidar2img = cam2img @ lidar2cam

            post_rot = img_aug_mats[i][:3, :3]
            post_tran = img_aug_mats[i][:3, 3]

            pts_img = (
                pts_lidar.tensor[:, :3] @ lidar2img[:3, :3].T +
                lidar2img[:3, 3])
            pts_img = torch.cat(
                [pts_img[:, :2] / pts_img[:, 2:3], pts_img[:, 2:3]], 1)
            pts_img = pts_img @ post_rot.T + post_tran

            depth_map = self.points2depth(pts_img, imgs[i].shape[0],
                                          imgs[i].shape[1])
            depth.append(depth_map)
        results['gt_depth'] = torch.stack(depth)
        return results


@TRANSFORMS.register_module()
class LoadOccFromFile(BaseTransform):

    def transform(self, results):
        occ_path = os.path.join(results['occ_path'], 'labels.npz')
        occ_labels = np.load(occ_path)

        results['gt_semantic_seg'] = occ_labels['semantics']
        results['mask_lidar'] = occ_labels['mask_lidar']
        results['mask_camera'] = occ_labels['mask_camera']
        return results


@TRANSFORMS.register_module()
class ImageAug3D(BaseTransform):

    def __init__(self,
                 final_dim,
                 resize_lim,
                 bot_pct_lim=[0.0, 0.0],
                 rot_lim=[0.0, 0.0],
                 rand_flip=False,
                 is_train=False):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results):
        H, W = results['ori_shape']
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(self, img, rotation, translation, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = Image.fromarray(img.astype('uint8'), mode='RGB')
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ])
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def transform(self, data):
        imgs = data['img']
        new_imgs = []
        transforms = []
        for img in imgs:
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                data)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(np.array(new_img).astype(np.float32))
            transforms.append(transform.numpy())
        data['img'] = new_imgs
        # update the calibration matrices
        data['img_aug_mat'] = transforms
        return data


@TRANSFORMS.register_module()
class BEVDataAug(BaseTransform):

    def __init__(self,
                 rot_lim=[0.0, 0.0],
                 scale_lim=[1.0, 1.0],
                 rand_flip=False):
        self.rot_lim = rot_lim
        self.scale_lim = scale_lim
        self.rand_flip = rand_flip

    def sample_augmentation(self):
        rotate = np.random.uniform(*self.rot_lim)
        scale = np.random.uniform(*self.scale_lim)
        flip_x = False
        flip_y = False
        if self.rand_flip:
            flip_x = np.random.choice([0, 1])
            flip_y = np.random.choice([0, 1])
        return rotate, scale, flip_x, flip_y

    def bev_transform(self, rotate, scale, flip_x, flip_y):
        theta = rotate / 180 * np.pi
        rotation = torch.Tensor([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
        scale_mat = torch.Tensor([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        if flip_x:
            flip_mat[0, 0] *= -1
        if flip_y:
            flip_mat[1, 1] *= -1
        rotation = flip_mat @ scale_mat @ rotation
        return rotation

    def transform(self, data):
        rotate, scale, flip_x, flip_y = self.sample_augmentation()
        assert rotate == 0 and scale == 1
        rotation = self.bev_transform(rotate, scale, flip_x, flip_y)

        if 'gt_semantic_seg' in data and (flip_x or flip_y):
            for key in ('gt_semantic_seg', 'mask_lidar', 'mask_camera'):
                if flip_x:
                    data[key] = data[key][::-1].copy()
                if flip_y:
                    data[key] = data[key][:, ::-1].copy()
        data['bev_aug_mat'] = rotation.numpy()
        return data


@TRANSFORMS.register_module()
class LoadFeatMaps(BaseTransform):

    def __init__(self, data_root, key, apply_aug=False):
        self.data_root = data_root
        self.key = key
        self.apply_aug = apply_aug

    def transform(self, results):
        feats = []
        img_aug_mats = results.get('img_aug_mat')
        for i, filename in enumerate(results['filename']):
            feat = np.load(
                os.path.join(self.data_root,
                             filename.split('/')[-1].split('.')[0] + '.npy'))
            feat = torch.from_numpy(feat)

            if self.apply_aug and img_aug_mats is not None:
                post_rot = img_aug_mats[i][:3, :3]
                post_tran = img_aug_mats[i][:3, 3]
                assert post_rot[0, 1] == post_rot[1, 0] == 0  # noqa

                h, w = feat.shape
                mode = 'nearest' if torch.all(feat == feat.floor()) else 'bilinear'
                feat = F.interpolate(
                    feat[None, None], (int(h * post_rot[1, 1] + 0.5),
                                       int(w * post_rot[0, 0] + 0.5)),
                    mode=mode).squeeze()
                feat = feat[int(post_tran[1]):, int(-post_tran[0]):]
            feats.append(feat)

        results[self.key] = torch.stack(feats)
        return results
