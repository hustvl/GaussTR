import os
import random
from collections import deque
from typing import Deque, Iterable

from mmdet3d.datasets import NuScenesDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class NuScenesOccDataset(NuScenesDataset):

    METAINFO = {
        'classes':
        ('others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation', 'free'),
    }

    def __init__(self,
                 metainfo=None,
                 load_adj_frame=False,
                 interval=1,
                 **kwargs):
        if not metainfo:
            metainfo = self.METAINFO
        elif 'classes' not in metainfo:
            metainfo['classes'] = self.METAINFO['classes']
        metainfo['label2cat'] = {
            i: cat_name
            for i, cat_name in enumerate(metainfo['classes'])
        }
        super().__init__(metainfo=metainfo, **kwargs)

        self.load_adj_frame = load_adj_frame
        self.interval = interval

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        get_data_info = super(NuScenesOccDataset, self).get_data_info
        input_dict = get_data_info(index)

        def get_curr_token(seq):
            curr_index = min(len(seq) - 1, 1)
            return seq[curr_index]['scene_token']

        def fetch_prev(seq: Deque, index, interval=1):
            if index == 0:
                return None
            prev = get_data_info(index - interval)
            if prev['scene_token'] != get_curr_token(seq):
                return None
            seq.appendleft(prev)
            return prev

        def fetch_next(seq: Deque, index, interval=1):
            if index >= len(self) - interval:
                return None
            next = get_data_info(index + interval)
            if next['scene_token'] != get_curr_token(seq):
                return None
            seq.append(next)
            return next

        if self.load_adj_frame:
            input_seq = deque([input_dict], maxlen=3)
            interval = random.randint(*self.interval) if isinstance(
                self.interval, Iterable) else self.interval
            if not fetch_prev(input_seq, index, interval):
                fetch_next(input_seq, index, interval)
                index += interval
            if not fetch_next(input_seq, index, interval):
                fetch_prev(input_seq, index - interval)

            assert (len(input_seq) == 3 and input_seq[0]['scene_token'] ==
                    input_seq[1]['scene_token'] == input_seq[2]['scene_token'])
            input_dict = self.concat_adj_frames(*input_seq)
        input_dict['occ_path'] = os.path.join(
            self.data_root,
            f"gts/{input_dict['scene_idx']}/{input_dict['token']}")
        return input_dict

    def concat_adj_frames(self, prev, curr, next=None):
        curr['images'] = dict(
            **curr['images'], **{
                f'PREV_{k}': v
                for k, v in prev['images'].items()
            })
        curr['ego2global'] = [curr['ego2global'], prev['ego2global']]

        if next is not None:
            curr['images'] = dict(
                **curr['images'], **{
                    f'NEXT_{k}': v
                    for k, v in next['images'].items()
                })
            curr['ego2global'].append(next['ego2global'])
        return curr
