import torch


from gaussion_point.getpointcloud import get_pointcloud
from gaussion_point.initializeparams import initializeparams
from utils.recon_helpers import setup_camera


def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, mean_sq_dist_method, densify_dataset=None):
    # Get RGB-D Data & Camera Parameters
    # A.数据获取:从数据集中获取第一帧RGB-D数据（颜色、深度）、相机内参和相机位姿
    color, depth, intrinsics, pose = dataset[0]

    # B.数据处理、格式处理、各项设置
    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    # C.密集化处理:如果传参提供了密集化数据集，则做相应处理
    if densify_dataset is not None:
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    # D.初始化点云和初始化参数，重点函数get_pointcloud()和initialize_params()
    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params, variables = initializeparams(init_pt_cld, num_frames, mean3_sq_dist)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam