import glob
import math
import pickle

import numpy as np
import open3d as o3d

COLOR_MAP = np.array([
    [0, 0, 0, 255],  # 0 undefined
    [47, 79, 79, 255],  # 1 barrier (sign)  Dark slategrey
    [255, 61, 99, 255],  # 2 bicycle  Red
    [220, 20, 60, 255],  # 3 bus (cyclist)  Crimson
    [255, 158, 0, 255],  # 4 car  Orange
    [255, 69, 0, 255],  # 5 cons. veh. (traiffic light)  Orangered
    [112, 128, 144, 255],  # 6 motorcycle  Slategrey
    [0, 0, 230, 255],  # 7 pedestrian  Blue
    [233, 150, 70, 255],  # 8 traffic cone  Darksalmon
    [255, 69, 0, 255],  # 9 trailer (traiffic light)  Orangered
    [165, 42, 42, 255],  # 10 trunk  nuTonomy green
    [0, 207, 191, 255],  # 11 driveable surface
    [115, 10, 67, 255],  # 12 other flat
    [75, 0, 75, 255],  # 13 sidewalk
    [112, 180, 60, 255],  # 14 terrain
    [222, 184, 135, 255],  # 15 manmade  Burlywood
    [0, 175, 0, 255],  # 16 vegetation  Green
])
COLOR_MAP = COLOR_MAP[:, :3] / 255


def voxel2points(voxels, voxel_size, pc_range, mask=None, ignore_labels=None):
    if mask is None:
        mask = np.ones_like(voxels)
    if ignore_labels is not None:
        for lbl in ignore_labels:
            mask &= voxels != lbl

    indices = np.where(mask)
    points = np.concatenate([
        indices[i][:, None] * voxel_size + voxel_size / 2 + pc_range[i]
        for i in range(len(indices))
    ], 1)
    return points, voxels[indices]


def voxel_profile(voxel, voxel_size):
    centers = np.concatenate((voxel[:, :2], voxel[:, 2:3] - voxel_size / 2),
                             axis=1)
    wlh = np.ones_like(centers) * voxel_size
    yaw = np.zeros_like(centers[:, 0:1])
    return np.concatenate((centers, wlh, yaw), axis=1)


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def compute_box_3d(center, size, heading_angle):
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    heading_angle = -heading_angle - math.pi / 2
    center[:, 2] = center[:, 2] + h / 2
    # R = rotz(1 * heading_angle)
    l, w, h = map(lambda x: (x / 2)[:, None], (l, w, h))

    x_corners = np.concatenate([-l, l, l, -l, -l, l, l, -l], axis=1)[..., None]
    y_corners = np.concatenate([w, w, -w, -w, w, w, -w, -w], axis=1)[..., None]
    z_corners = np.concatenate([h, h, h, h, -h, -h, -h, -h], axis=1)[..., None]
    # corners_3d = R @ torch.vstack([x_corners, y_corners, z_corners])
    corners_3d = np.concatenate([x_corners, y_corners, z_corners], axis=2)
    corners_3d += center[:, None]
    return corners_3d


def generate_ego_car():
    ego_range = [-2, -1, 0, 2, 1, 1.5]
    ego_voxel_size = [0.1, 0.1, 0.1]
    ego_xdim, ego_ydim, ego_zdim = map(
        lambda x: int((ego_range[x + 3] - ego_range[x]) / ego_voxel_size[x]),
        range(3))

    ego_points = np.stack(
        np.meshgrid(*map(np.arange, (ego_xdim, ego_ydim, ego_zdim))),
        axis=-1).reshape(-1, 3)
    ego_points = np.concatenate(
        list(
            map(
                lambda x: (ego_points[:, x:x + 1] + 0.5) / ego_xdim *
                (ego_range[x + 3] - ego_range[x]) + ego_range[x], range(3))),
        axis=-1)

    ego_points_label = (np.ones((ego_points.shape[0])) * 16).astype(np.uint8)
    ego_dict = {}
    ego_dict['point'] = ego_points
    ego_dict['label'] = ego_points_label
    return ego_points


def show_point_cloud(points,
                     points_colors=None,
                     voxel_size=0.4,
                     vertices=None,
                     edges=None,
                     show_coord_frame=False,
                     show_ego_car=True):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if points_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(points_colors[:, :3])

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=voxel_size)
    vis.add_geometry(voxel_grid)

    if vertices is not None:
        linesets = o3d.geometry.LineSet()
        linesets.points = o3d.open3d.utility.Vector3dVector(
            vertices.reshape((-1, 3)))
        linesets.lines = o3d.open3d.utility.Vector2iVector(
            edges.reshape((-1, 2)))
        linesets.paint_uniform_color((0, 0, 0))
        vis.add_geometry(linesets)

    if show_coord_frame:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.6, origin=[0, 0, 0])
        vis.add_geometry(coord_frame)

    if show_ego_car:
        ego_pcd = o3d.geometry.PointCloud()
        ego_pcd.points = o3d.utility.Vector3dVector(generate_ego_car())
        vis.add_geometry(ego_pcd)
    return vis


def visualize_occ(voxels,
                  voxel_size,
                  pcd_range,
                  ignore_labels=None,
                  save=False,
                  show_voxel_bbox=True,
                  **kwargs):
    points, labels = voxel2points(
        voxels, voxel_size, pcd_range, ignore_labels=ignore_labels)
    pcd_colors = COLOR_MAP[labels]

    if show_voxel_bbox:
        bboxes = voxel_profile(points, voxel_size)
        vertices = compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])
        bases_ = np.arange(0, vertices.shape[0] * 8, 8)
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
                          [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
        edges = np.tile(edges[np.newaxis], (vertices.shape[0], 1, 1))
        edges = edges + bases_[:, None, None]
        kwargs['vertices'] = vertices
        kwargs['edges'] = edges

    vis = show_point_cloud(
        points=points,
        points_colors=pcd_colors,
        voxel_size=voxel_size,
        **kwargs)

    # view_control = vis.get_view_control()
    # view_control.set_zoom(args.zoom)
    # view_control.set_up(args.up_vec)
    # view_control.set_front(args.front_vec)
    # view_control.set_lookat(np.array([points.mean(axis=0)[0], 0, 0]))
    # vis.poll_events()
    # vis.update_renderer()
    vis.run()
    if save:
        vis.capture_screen_image(f'outputs/{save}.png')
    vis.destroy_window()
    del vis


def main():
    path = './0.pkl'

    kwargs = {
        'voxel_size': 0.4,
        'pcd_range': [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        'ignore_labels': [17, 255]
    }

    for file in glob.glob(path):
        with open(file, 'rb') as f:
            outputs = pickle.load(f)
        for occ in (outputs['occ_pred'], outputs['occ_gt']):
            visualize_occ(occ, **kwargs)


if __name__ == '__main__':
    main()
