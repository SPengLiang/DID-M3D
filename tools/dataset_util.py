import numpy as np
import cv2
import pathlib
import re
import open3d as o3d
import xml.etree.ElementTree as ET

from lib.datasets.kitti_utils import Calibration, Object3d, get_objects_from_label


class Dataset:
    def __init__(self, split, dataset_path):
        assert split in ["train", "val", "test"]
        self.split = split
        self.dataset_name = 'testing' if split == 'test' else 'training'
        self.dataset_path = pathlib.Path(rf"{dataset_path}/{self.dataset_name}")
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def get_lidar(self, idx):
        lidar_file = self.dataset_path / 'velodyne' / ('%06d.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_instance(self, idx):
        instance_file = self.dataset_path / 'instance_2' / ('%06d.png' % idx)
        assert instance_file.exists()
        return cv2.imread(str(instance_file))

    def get_patchwork(self, idx):
        patchwork_file = self.dataset_path / 'patchwork' / ('%06d.label' % idx)
        assert patchwork_file.exists()
        return np.fromfile(str(patchwork_file), dtype=np.int32).reshape(-1, 1)

    def get_lidar_with_ground(self, idx, fov=False):
        lidar = self.get_lidar(idx)[:,:3]
        patchwork = self.get_patchwork(idx).reshape(-1)
        calib = self.get_calib(idx)
        lidar = calib.lidar_to_rect(lidar[:, :3])
        if fov:
            shape = self.get_image(idx).shape
            flag = self.get_fov_flag(lidar, shape, calib)
            lidar = lidar[flag]
            patchwork = patchwork[flag]
        ground = lidar[patchwork == 1]
        non_ground = lidar[patchwork != 1]
        return ground, non_ground

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_plane(self, idx):
        plane_file = self.dataset_path / 'planes' / ('%06d.txt' % idx)
        assert plane_file.exists()

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    def get_blank(self, idx):
        blank_file = self.dataset_path / 'blank' / ('%06d.png' % idx)
        assert blank_file.exists()
        return cv2.imread(str(blank_file))

    def get_blank_origin(self, idx):
        blank_file = self.dataset_path / 'blank_2' / ('%06d.png' % idx)
        label_file = self.dataset_path / 'blank_2' / ('%06d.xml' % idx)
        if not label_file.exists():
            return None, None
        assert blank_file.exists()
        attrib = ET.parse(str(label_file)).getroot().attrib
        verified = attrib.get('verified', 'no')
        if verified == 'no':
            return None, None

        img = cv2.imread(str(blank_file))
        h, w, _ = img.shape
        masked, blank = img[:h // 2, :], img[h // 2:, :]
        mask = (blank - masked == 0).any(axis=2).astype(np.uint8) * 255
        mask = cv2.erode(mask, self.kernel, iterations=1)
        return mask, blank

    def get_lidar(self, idx):
        lidar_file = self.dataset_path / 'velodyne' / ('%06d.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_image(self, idx):
        image_file = self.dataset_path / 'image_2' / ('%06d.png' % idx)
        assert image_file.exists()
        return cv2.imread(str(image_file))

    def get_calib(self, idx):
        calib_file = self.dataset_path / 'calib' / ('%06d.txt' % idx)
        assert calib_file.exists()
        return Calibration(str(calib_file))

    def get_depth_penet(self, idx):
        depth_file = self.dataset_path / 'depth_penet' / ('%06d.png' % idx)
        assert depth_file.exists()
        return cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED) / 256.0

    def get_depth(self, idx):
        depth_file = self.dataset_path / 'depth_dense' / ('%06d.png' % idx)
        assert depth_file.exists()
        return cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED) / 256.0

    def get_blank_with_depth(self, idx):
        rgb = self.get_blank(idx)
        depth_file = self.dataset_path / 'depth_blank' / ('%06d.png' % idx)
        assert depth_file.exists()
        depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED) / 256.0

        h, w, _ = rgb.shape
        h_, w_ = depth.shape
        pad_h, pad_w = h - h_, w - w_
        depth = np.pad(depth, ((0, pad_h), (0, pad_w)), mode="edge")
        return rgb, depth

    def get_image_with_depth(self, idx, use_penet=False):
        rgb = self.get_image(idx)
        if use_penet:
            depth = self.get_depth_penet(idx)
            h, w, _ = rgb.shape
            h_, w_ = depth.shape
            pad_h, pad_w = h - h_, w - w_
            depth = np.pad(depth, ((0, pad_h), (0, pad_w)), mode="edge")
        else:
            depth = self.get_depth(idx)
            h, w, _ = rgb.shape
            h_, w_ = depth.shape
            pad_h, pad_wl = h - h_, (w - w_) // 2
            pad_wr = w - w_ - pad_wl
            depth = np.pad(depth, ((pad_h, 0), (pad_wl, pad_wr)), mode="edge")
        return rgb, depth

    def get_label(self, idx):
        label_file = self.dataset_path / 'label_2' / ('%06d.txt' % idx)
        assert label_file.exists()
        return get_objects_from_label(label_file)

    def get_bbox3d_rect(self, idx, on_button=False):
        obj_list = self.get_label(idx)
        num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])

        annos = dict()
        annos['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
        annos['location'] = np.concatenate([obj.pos.reshape(1, 3) for obj in obj_list], axis=0)
        annos['rotation_y'] = np.array([obj.ry for obj in obj_list])

        pos = annos['location'][:num_objects]
        dims = annos['dimensions'][:num_objects]
        rots = annos['rotation_y'][:num_objects]
        l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]

        if on_button:
            pass
        else:
            pos[:, 1] -= h[:, 0] / 2

        bbox3d = np.concatenate([pos, l, h, w, rots[..., np.newaxis]], axis=1)
        return bbox3d

    def get_bbox2d(self, idx, chosen_cls=('Car', 'Pedestrian', 'Cyclist'), from_box3d=False):
        obj_list = self.get_label(idx)
        if from_box3d:
            raise NotImplementedError
        else:
            bbox2d = np.array([obj.box2d for obj in obj_list if obj.cls_type in chosen_cls])
            bbox2d = np.round(bbox2d).astype(int)
            cls = [obj.cls_type for obj in obj_list if obj.cls_type in chosen_cls]
            label = [obj for obj in obj_list if obj.cls_type in chosen_cls]
        return bbox2d, cls, label

    def get_bbox(self, idx, chosen_cls=('Car', 'Pedestrian', 'Cyclist')):
        obj_list = self.get_label(idx)
        obj_list = [obj for obj in obj_list if obj.cls_type in chosen_cls]
        bbox3d = np.array([[obj.pos[0], obj.pos[1], obj.pos[2], obj.l, obj.h, obj.w, obj.ry] for obj in obj_list])
        bbox2d = np.array([obj.box2d for obj in obj_list])
        return bbox3d, bbox2d, obj_list
    @staticmethod
    def split_dict(data):
        length = len(next(iter(data.values())))
        result = [{} for i in range(length)]
        for key, value in data.items():
            for i in range(length):
                result[i][key] = value[i]

        return result
