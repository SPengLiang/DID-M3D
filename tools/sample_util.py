import numpy as np
import open3d as o3d
import pathlib
import re
import cv2
import pickle
from lib.datasets.kitti_utils import Calibration
from tools.dataset_util import Dataset
from sklearn.decomposition import PCA
from tools.box_util import boxes_bev_iou_cpu, rect2lidar, check_points_in_boxes3d
from lib.datasets.kitti_utils import Object3d


def merge_labels(labels, samples, calib_, image_shape):
    assert all([label.cls_type == "Car" for label in labels])
    canvas = np.zeros(image_shape[:2], dtype=np.int8) - 1
    labels += [sample.to_label() for sample in samples]
    labels = sorted(labels, key=lambda x: x.pos[2], reverse=True)
    for i, label in enumerate(labels):
        corners = label.generate_corners3d()
        uv, _ = calib_.rect_to_img(corners)
        u_min = round(max(0, np.min(uv[:, 0])))
        v_min = round(max(0, np.min(uv[:, 1])))
        u_max = round(min(np.max(uv[:, 0]), image_shape[1]))
        v_max = round(min(np.max(uv[:, 1]), image_shape[0]))

        canvas[v_min: v_max, u_min: u_max] = i
        label.area = (v_max - v_min) * (u_max - u_min)
    for i, label in enumerate(labels):
        area = np.sum(canvas == i)
        label.area = 1 - area / label.area
        label.occlusion = area2occlusion(label.area)
    return labels

def area2occlusion(area):
    if area < 0.1:
        return 0
    elif area < 0.3:
        return 1
    elif area < 0.5:
        return 2
    else:
        return 3

def to3d(image, depth, calib, bbox2d=None):
    assert image.shape[:2] == depth.shape
    h, w = depth.shape
    u = np.repeat(np.arange(w), h)
    v = np.tile(np.arange(h), w)
    d = depth[v, u]
    rgb = image[v, u][:, ::-1]
    if bbox2d:  # 对样本，需要附加其在图像的位置
        u += bbox2d[0]
        v += bbox2d[1]
    cord = calib.img_to_rect(u, v, d)
    return cord, rgb


def to2d(cord, rgb, calib):
    uv, d = calib.rect_to_img(cord[:, 0:3])
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    # 图像大小可能与原图不一致
    width, height = u.max() + 1, v.max() + 1

    image = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.float32)
    image[v, u] = rgb
    depth[v, u] = d
    return image, depth


class SampleDatabase:
    def __init__(self, database_path):
        self.database_path = pathlib.Path(database_path)
        assert self.database_path.exists()
        self.image_path = self.database_path / "image"
        self.depth_path = self.database_path / "depth"
        self.mask_path = self.database_path / "mask"
        with open(self.database_path / "kitti_car_database.pkl", "rb") as f:
            database = pickle.load(f)
        self.database = list(database.values())

        self.sample_group = {
            "sample_num": 20,
            "pointer": len(database),
            "x_range": [[-15.], [15.]],
            "z_range": [[20.], [70.]],
            "indices": None
        }

    @staticmethod
    def get_ry_(xyz, xyz_, ry, calib, calib_):
        uv, _ = calib.rect_to_img(xyz.reshape(1, -1))
        alpha = calib.ry2alpha(ry, uv[:, 0])
        uv_, _ = calib_.rect_to_img(xyz_.reshape(1, -1))
        ry_ = calib_.alpha2ry(alpha, uv_[:, 0])
        return ry_

    @staticmethod
    def get_y_on_plane(x, z, plane):
        a, b, c, d = plane
        y = - a * x - c * z - d
        y /= b
        return y

    def sample_with_fixed_number(self, calib_):
        database, sample_group = self.database, self.sample_group
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        low_x, high_x = sample_group["x_range"]
        low_z, high_z = sample_group["z_range"]
        if pointer >= len(database):
            indices = np.random.permutation(len(database))
            pointer = 0

        samples = [database[idx] for idx in indices[pointer: pointer + sample_num]]

        # 获取原始 bbox3d
        xyz = np.array([s['label'].pos for s in samples])
        ry = np.array([[s['label'].ry] for s in samples])
        lhw = np.array([[s['label'].l, s['label'].h, s['label'].w] for s in samples])
        calib = [s['calib'] for s in samples]
        plane = [s['plane'] for s in samples]

        # 采样 bbox3d
        x_ = np.random.uniform(low=low_x, high=high_x, size=(sample_num, 1))
        z_ = np.random.uniform(low=low_z, high=high_z, size=(sample_num, 1))
        y_ = np.array([self.get_y_on_plane(x_[i], z_[i], plane[i]) for i in range(sample_num)])
        xyz_ = np.concatenate([x_, y_, z_], axis=1)
        ry_ = np.array([self.get_ry_(xyz[i], xyz_[i], ry[i], calib[i], calib_) for i in range(sample_num)])
        bbox3d_ = np.concatenate([xyz_, lhw, ry_], axis=1)

        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return samples, bbox3d_

    @staticmethod
    def check_normal_angle(normal, max_degree):
        assert normal.shape[0] == 3
        limit = np.cos(np.radians(max_degree))
        norm = np.linalg.norm(normal)
        cos = np.abs(normal[1]) / norm  # abs: 法向量不一定向下
        return cos >= limit

    def sample_put_on_plane(self, bbox3d, ground, radius=2, min_num=10, max_var=0.5e-2, max_degree=20):
        bbox3d = bbox3d.copy()
        flag = np.zeros((bbox3d.shape[0]), dtype=bool)
        for i, pos in enumerate(bbox3d[:, :3]):
            distance = np.linalg.norm(ground - pos, axis=1)
            nearby = ground[distance < radius]
            if nearby.shape[0] < min_num:
                continue

            pca = PCA(n_components=3)
            pca.fit(nearby)
            normal = pca.components_[2]
            var = pca.explained_variance_ratio_[2]
            if var > max_var:
                continue
            if not self.check_normal_angle(normal, max_degree):
                continue
            d = -normal.dot(np.mean(nearby, axis=0))
            bbox3d[i, 1] = self.get_y_on_plane(pos[0], pos[2], [*normal, d])
            flag[i] = True
        return bbox3d, flag

    def get_samples(self, ground, non_ground, calib):
        samples, bbox3d = self.sample_with_fixed_number(calib)

        # 放置于地面，第一次筛除
        bbox3d_, flag1 = self.sample_put_on_plane(bbox3d, ground)
        if flag1.sum() == 0:
            return []

        # api 要求 bbox3d 为 lidar 坐标系
        bbox3d_in_lidar = rect2lidar(bbox3d_[flag1], calib)

        # 判断样本间是否有重叠，第二次筛除
        iou = boxes_bev_iou_cpu(bbox3d_in_lidar, bbox3d_in_lidar)
        iou[range(bbox3d_in_lidar.shape[0]), range(bbox3d_in_lidar.shape[0])] = 0
        flag2 = iou.max(axis=1) == 0
        if flag2.sum() == 0:
            return []

        # 判断样本是否与障碍物重叠，第三次筛除
        points_in_lidar = calib.rect_to_lidar(non_ground)
        flag3 = ~ check_points_in_boxes3d(points_in_lidar, bbox3d_in_lidar[flag2])
        if flag3.sum() == 0:
            return []

        # 合并筛除结果
        valid = np.arange(bbox3d.shape[0])[flag1][flag2][flag3]
        res = [Sample(samples[i], bbox3d_[i], calib, self) for i in valid]
        return res

    @staticmethod
    def add_samples_to_scene(samples, image, depth):
        image_, depth_ = image.copy(), depth.copy()
        flag = np.zeros(len(samples), dtype=bool)
        for i, sample in enumerate(samples):
            image_, depth_, flag[i] = sample.cover(image_, depth_)

        return image_, depth_, [sample for i, sample in enumerate(samples) if flag[i]]


class Sample:
    def __init__(self, sample, bbox3d, calib, database: SampleDatabase):
        self.sample = sample
        self.bbox3d_ = bbox3d
        self.database = database

        self.label = sample['label']
        self.calib = sample['calib']
        self.calib_ = calib
        self.plane = sample['plane']
        self.bbox2d = sample['bbox2d']
        self.name = sample['name']

        self.image = self.get_image()
        self.depth = self.get_depth()

        self.occlusion_ = 0  # 需要在最终图像中求
        self.image_, self.depth_, self.bbox2d_ = self.transform()

    def __repr__(self):
        return f"Sample(name={self.name})"

    def get_image(self):
        image_file = self.database.image_path / (self.name + ".png")
        assert image_file.exists()
        return cv2.imread(str(image_file))

    def get_depth(self):
        depth_file = self.database.depth_path / (self.name + ".png")
        assert depth_file.exists()
        return cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED) / 256.0

    def get_points(self):
        assert self.depth.shape[:2] == self.image.shape[:2]
        image, depth, calib, label = self.image, self.depth, self.calib, self.label
        calib_, bbox3d_, bbox2d = self.calib_, self.bbox3d_, self.bbox2d
        xyz, ry = label.pos, label.ry
        xyz_, ry_ = bbox3d_[:3], bbox3d_[6]

        cord, rgb = to3d(image, depth, calib, bbox2d)

        # 删除 d = 0 的点
        valid = cord[:, 2] >= 1e-3
        cord, rgb = cord[valid], rgb[valid]

        cord = cord - xyz
        dr = ry_ - ry
        R = np.array([[np.cos(dr), 0, np.sin(dr)],
                      [0, 1, 0],
                      [-np.sin(dr), 0, np.cos(dr)]])
        cord = cord @ R.T + xyz_

        return cord, rgb

    @staticmethod
    def get_3d_center_in_2d(xyz, calib):
        xyz = xyz.reshape(1, -1)[:, :3]
        uv, _ = calib.rect_to_img(xyz)
        uv = np.round(uv).astype(int).reshape(2)
        return uv

    def transform(self):
        assert self.depth.shape[:2] == self.image.shape[:2]
        image, depth, calib, label = self.image, self.depth, self.calib, self.label
        calib_, bbox3d_, bbox2d = self.calib_, self.bbox3d_, self.bbox2d

        center = self.get_3d_center_in_2d(label.pos, self.calib)
        center_ = self.get_3d_center_in_2d(bbox3d_[:3], calib_)
        dry = bbox3d_[6] - label.ry  # ry_ - ry
        h, w = depth.shape

        offset = np.arange(w, dtype=int) - center[0] + bbox2d[0]
        offset = - np.tan(dry) * offset * label.w / w

        depth_ = depth - label.pos[2] + offset.reshape(1, -1) + bbox3d_[2]
        depth_[depth < 1e-2] = 0

        rate = bbox3d_[2] / label.pos[2]  # z_ / z
        h_, w_ = round(h / rate), round(w / rate)

        depth_ = cv2.resize(depth_, (w_, h_), interpolation=cv2.INTER_NEAREST)
        image_ = cv2.resize(image, (w_, h_), interpolation=cv2.INTER_NEAREST)

        bbox2d_ = np.tile((bbox2d[:2] - center) / rate + center_, 2)
        bbox2d_ = np.round(bbox2d_).astype(int)
        # 这里 bbox2d_ 是可能在背景图像外的
        bbox2d_[2:] += [w_, h_]  # 避免 bbox2d_ 与 image_ 的大小不一致

        return image_, depth_, bbox2d_.tolist()

    def cover(self, image, depth, area_threshold=0.5):
        assert image.shape[:2] == depth.shape
        blank_rgb, blank_d = image.copy(), depth.copy()
        image_, depth_, bbox2d_ = self.image_, self.depth_, self.bbox2d_

        u_min, v_min, u_max, v_max = bbox2d_
        # 避免 bbox2d_ 在图像外
        if u_min < 0 or v_min < 0 or u_max > image.shape[1] or v_max > image.shape[0]:
            return blank_rgb, blank_d, False

        d_in_bbox2d = blank_d[v_min: v_max, u_min: u_max]
        valid = (depth_ > 1e-2) & (depth_ < d_in_bbox2d)
        area = (v_max - v_min) * (u_max - u_min) - np.sum(depth_ <= 1e-2)
        valid_rate = np.sum(valid) / area
        if valid_rate <= area_threshold:
            return blank_rgb, blank_d, False

        blank_rgb[v_min: v_max, u_min: u_max][valid] = image_[valid]
        blank_d[v_min: v_max, u_min: u_max][valid] = depth_[valid]

        return blank_rgb, blank_d, True

    def to_label(self):
        label = self.label
        cls = label.cls_type
        trucation = 0
        score = 0
        occlusion = 0
        x_, y_, z_, l_, h_, w_, ry_ = self.bbox3d_
        alpha = self.get_alpha(self.bbox3d_[:3], ry_, self.calib_)
        u_min, v_min, u_max, v_max = self.bbox2d_
        line = f"{cls} {trucation} {occlusion} {alpha} {u_min} {v_min} {u_max} {v_max} {h_} {w_} {l_} {x_} {y_} {z_} {ry_} {score}"
        return Object3d(line)

    @staticmethod
    def get_alpha(xyz, ry, calib):
        uv, _ = calib.rect_to_img(xyz.reshape(1, -1))
        alpha = calib.ry2alpha(ry, uv[:, 0])[0]
        return alpha


from pathlib import Path

if __name__ == '__main__':
    test_dir = Path("/mnt/e/DataSet/kitti/kitti_img_database/test")
    np.random.seed(0)

    database = SampleDatabase("/mnt/e/DataSet/kitti/kitti_img_database/")
    dataset = Dataset("train", r"/mnt/e/DataSet/kitti")

    for idx in range(200):
        calib_ = dataset.get_calib(idx)
        image, depth = dataset.get_image_with_depth(idx, use_penet=True)
        ground, non_ground = dataset.get_lidar_with_ground(idx, fov=True)
        samples = database.get_samples(ground, non_ground, calib_)
        image_, depth_, flag = database.add_samples_to_scene(samples, image, depth)
        cv2.imwrite(str(test_dir / ('%06d.png' % idx)), image_)
