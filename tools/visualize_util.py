import open3d as o3d
import numpy as np


def generate_corners(bbox3d):
    corners = []
    bbox3d = bbox3d.reshape(-1, 7)
    for box in bbox3d:
        xyz, l, h, w, ry = box[0:3], box[3], box[4], box[5], box[6]
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(ry), 0, np.sin(ry)],
                      [0, 1, 0],
                      [-np.sin(ry), 0, np.cos(ry)]])

        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners.append(corners3d + xyz)
    res = np.concatenate(corners, axis=0).reshape(-1, 8, 3)
    return res

def draw_3d_box(corner, color=(1, 0, 0)):
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(corner),
                                    lines=o3d.utility.Vector2iVector(lines))
    line_set.paint_uniform_color(color)
    return line_set


def show_o3d(cord, rgb=None, bbox3d=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cord[:, 0:3])
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb / 256.)
    else:
        pcd.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]) / 256.)

    if bbox3d is not None:
        corners = generate_corners(bbox3d)
        for corner in corners:
            line_set = draw_3d_box(corner)
            vis.add_geometry(line_set)

    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()
