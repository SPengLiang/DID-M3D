import numpy as np
import torch
import copy
from aug.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
from aug.iou3d_nms.iou3d_nms_utils import boxes_bev_iou_cpu
from tools.common_util import check_numpy_to_torch


def rect2lidar(bbox3d, calib):
    loc, lhw, ry = bbox3d[:, 0:3], bbox3d[:, 3:6], bbox3d[:, 6:]
    loc = calib.rect_to_lidar(loc)
    loc[:, 2] += lhw[:, 1] / 2  # 现在是 lidar 坐标系，所以是 +
    lwh = lhw[:, [0, 2, 1]]
    rz = -ry - np.pi / 2
    return np.concatenate([loc, lwh, rz], axis=1)


def remove_points_in_boxes3d(points, boxes3d, enlarge=False):
    """
    Args:
        points: (num_points, 3 + C)
        boxes3d: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps
        enlarge: float or False
    Returns:

    """
    points = copy.deepcopy(points)
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    points, is_numpy = check_numpy_to_torch(points)
    if enlarge:
        boxes3d[:, 3:6] += enlarge
    point_masks = points_in_boxes_cpu(points[:, 0:3], boxes3d)
    points = points[point_masks.sum(dim=0) == 0]

    return points.numpy() if is_numpy else points


def get_objects_in_boxes3d(points, boxes3d, enlarge=False):
    points = copy.deepcopy(points)
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    points, is_numpy = check_numpy_to_torch(points)
    objects = []
    if enlarge:
        boxes3d[:, 3:6] += enlarge
    for box3d in boxes3d:
        point_masks = points_in_boxes_cpu(points[:, 0:3], box3d.unsqueeze(dim=0))
        obj = points[point_masks.sum(dim=0) > 0]
        objects.append(obj.numpy()) if is_numpy else obj

    return objects


def check_points_in_boxes3d(points, boxes3d, enlarge=False):
    points = copy.deepcopy(points)
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    points, is_numpy = check_numpy_to_torch(points)
    if enlarge:
        boxes3d[:, 3:6] += enlarge
    point_masks = points_in_boxes_cpu(points[:, 0:3], boxes3d)
    flag = point_masks.sum(dim=1) > 0
    return flag.numpy() if is_numpy else flag


def iou2d(box_a, box_b):
    """
    Args:
        box_a: (4) [x1, y1, x2, y2]
        box_b: (4) [x1, y1, x2, y2]
    """
    assert box_a[0] <= box_a[2] and box_a[1] <= box_a[3]
    assert box_b[0] <= box_b[2] and box_b[1] <= box_b[3]

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    area_inter = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])) * \
                 max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))

    return area_inter / (area_a + area_b - area_inter + 1e-6)

def boxes_iou2d(box_a, box_b):
    """
    Args:
        box_a: (N, 4) [x1, y1, x2, y2]
        box_b: (M, 4) [x1, y1, x2, y2]
    Returns:
        iou: (N, M)
    """
    assert box_a.shape[1] == box_b.shape[1] == 4
    n = box_a.shape[0]
    m = box_b.shape[0]
    iou = np.zeros((n, m), dtype=np.float32)

    for i in range(n):
        for j in range(m):
            iou[i, j] = iou2d(box_a[i], box_b[j])
    return iou

def find_best_match(iou, threshold=0.6):
    """
    Args:
        iou: (N, M)
        threshold:
    Returns:
        best_match: (N), index of matched box in b
    """
    best_match = iou.argmax(axis=1)
    best_match_value = iou.max(axis=1)
    best_match[best_match_value < threshold] = -1
    return best_match


if __name__ == '__main__':
    box1 = torch.tensor([1, 1, 4, 4])
    box2 = torch.tensor([2, 2, 5, 5])

    print(iou2d(box1, box2))
    print(iou2d(box2, box1))