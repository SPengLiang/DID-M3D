import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from aug import roiaware_pool3d_cuda


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def points_in_boxes_cpu(points, boxes):
    """
    Args:
        points: (num_points, 3)
        boxes: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps
    Returns:
        point_indices: (N, num_points)
    """
    assert boxes.shape[1] == 7
    assert points.shape[1] == 3
    points, is_numpy = check_numpy_to_torch(points)
    boxes, is_numpy = check_numpy_to_torch(boxes)

    point_indices = points.new_zeros((boxes.shape[0], points.shape[0]), dtype=torch.int)
    roiaware_pool3d_cuda.points_in_boxes_cpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)

    return point_indices.numpy() if is_numpy else point_indices



if __name__ == '__main__':
    pass
