/*
RoI-aware point cloud feature pooling
Reference paper:  https://arxiv.org/abs/1907.03670
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <assert.h>


inline void lidar_to_local_coords_cpu(float shift_x, float shift_y, float rot_angle, float &local_x, float &local_y){
    float cosa = cos(-rot_angle), sina = sin(-rot_angle);
    local_x = shift_x * cosa + shift_y * (-sina);
    local_y = shift_x * sina + shift_y * cosa;
}


inline int check_pt_in_box3d_cpu(const float *pt, const float *box3d, float &local_x, float &local_y){
    // param pt: (x, y, z)
    // param box3d: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    const float MARGIN = 1e-2;
    float x = pt[0], y = pt[1], z = pt[2];
    float cx = box3d[0], cy = box3d[1], cz = box3d[2];
    float dx = box3d[3], dy = box3d[4], dz = box3d[5], rz = box3d[6];

    if (fabsf(z - cz) > dz / 2.0) return 0;
    lidar_to_local_coords_cpu(x - cx, y - cy, rz, local_x, local_y);
    float in_flag = (fabs(local_x) < dx / 2.0 + MARGIN) & (fabs(local_y) < dy / 2.0 + MARGIN);
    return in_flag;
}


int points_in_boxes_cpu(at::Tensor boxes_tensor, at::Tensor pts_tensor, at::Tensor pts_indices_tensor){
    // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps
    // params pts: (num_points, 3) [x, y, z]
    // params pts_indices: (N, num_points)

//    CHECK_CONTIGUOUS(boxes_tensor);
//    CHECK_CONTIGUOUS(pts_tensor);
//    CHECK_CONTIGUOUS(pts_indices_tensor);

    int boxes_num = boxes_tensor.size(0);
    int pts_num = pts_tensor.size(0);

    const float *boxes = boxes_tensor.data<float>();
    const float *pts = pts_tensor.data<float>();
    int *pts_indices = pts_indices_tensor.data<int>();

    float local_x = 0, local_y = 0;
    for (int i = 0; i < boxes_num; i++){
        for (int j = 0; j < pts_num; j++){
            int cur_in_flag = check_pt_in_box3d_cpu(pts + j * 3, boxes + i * 7, local_x, local_y);
            pts_indices[i * pts_num + j] = cur_in_flag;
        }
    }

    return 1;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("points_in_boxes_cpu", &points_in_boxes_cpu, "points_in_boxes_cpu forward (CUDA)");
}
