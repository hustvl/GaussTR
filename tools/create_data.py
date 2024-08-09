import argparse
from os import path as osp

import mmengine

from nuscenes import NuScenes
from pathlib import Path


def update_nuscenes_occ_infos(root_path, pkl_path, out_dir):
    print(f'Updating occ infos for {pkl_path}.')
    data = mmengine.load(pkl_path)
    nusc = NuScenes(
        version=data['metainfo']['version'], dataroot=root_path, verbose=False)

    print('Start updating:')
    for i, info in enumerate(mmengine.track_iter_progress(data['data_list'])):
        sample = nusc.get('sample', info['token'])
        data['data_list'][i]['scene_token'] = sample['scene_token']
        scene = nusc.get('scene', sample['scene_token'])
        data['data_list'][i][
            'occ_path'] = f"gts/{scene['name']}/{info['token']}"

    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')

    mmengine.dump(data, out_path, 'pkl')


def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    # from tools.dataset_converters import nuscenes_converter as nuscenes_converter
    # from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos
    # nuscenes_converter.create_nuscenes_infos(
    #     root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    # update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_train_path)
    # update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_val_path)
    # create_groundtruth_database(dataset_name, root_path, info_prefix,
    #                             f'{info_prefix}_infos_train.pkl')
    update_nuscenes_occ_infos(root_path, info_train_path, out_dir)
    update_nuscenes_occ_infos(root_path, info_val_path, out_dir)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet3d')

    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
