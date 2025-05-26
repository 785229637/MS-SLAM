import cv2
import numpy as np
import torch
import torch.nn.functional as F


def compute_keyframe_union_mask(
    params: dict,
    dataset: object,
    keyframe_list: list,
    project_to_image: callable,
    world_to_camera: callable,
    quaternion_to_rotation_matrix: callable
) -> torch.Tensor:

    orig_width = dataset.orig_width
    orig_height = dataset.orig_height
    fx, fy, cx, cy = dataset.fx, dataset.fy, dataset.cx, dataset.cy


    device = params['means3D'].device
    total_mask = torch.zeros(len(params['means3D']), 
                     dtype=torch.bool, 
                     device=device)

    for time_idx in keyframe_list:

        cam_rot = quaternion_to_rotation_matrix(
            params['cam_unnorm_rots'][..., time_idx])
        cam_tran = params['cam_trans'][..., time_idx]

        camera_points = world_to_camera(params['means3D'], cam_rot, cam_tran)
        image_points = project_to_image(camera_points, fx, fy, cx, cy)

        current_mask = (
            (image_points[:, 0] >= 0) &           
            (image_points[:, 0] < orig_width) &     
            (image_points[:, 1] >= 0) &           
            (image_points[:, 1] < orig_height)       
        )

        total_mask |= current_mask

    return total_mask


def get_salient_regions(depth):

    depth = depth.unsqueeze(0).unsqueeze(0) if depth.dim() == 2 else depth
    

    sobel_x = torch.tensor([[-1, 0, 1], 
                           [-2, 0, 2], 
                           [--1, 0, 1]], dtype=torch.float32, device=depth.device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],
                           [ 0, 0, 0],
                           [ 1, 2, 1]], dtype=torch.float32, device=depth.device).view(1,1,3,3)


    grad_x = F.conv2d(depth, sobel_x, padding=1)
    grad_y = F.conv2d(depth, sobel_y, padding=1)
    

    grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
  
    threshold = grad_mag.mean() 
    salient_mask = (grad_mag > threshold).squeeze()
    
    return salient_mask

def create_point_mask(image_points, depth):

    salient_mask = get_salient_regions(depth)
    H, W = salient_mask.shape
    

    u = torch.round(image_points[:, 0]).long()
    v = torch.round(image_points[:, 1]).long()
    

    valid_u = (u >= 0.1*W) & (u < 0.9*W)
    valid_v = (v >= 0.1*H) & (v < 0.9*H)
    valid_mask = valid_u & valid_v
    

    point_mask = torch.zeros(image_points.shape[0], 
                            dtype=torch.bool,
                            device=image_points.device)
    

    point_mask[valid_mask] = salient_mask[v[valid_mask], u[valid_mask]]
    
    return point_mask

def compute_depth_gradient(depth_map):

    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    return gradient_magnitude
