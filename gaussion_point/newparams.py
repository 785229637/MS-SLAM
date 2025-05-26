
import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from gaussion_point.getpointcloud import get_pointcloud
from utils.slam_external import build_rotation
from utils.slam_helpers import transform_to_frame, transformed_params2depthplussilhouette


def initialize_new_params(new_pt_cld, mean3_sq_dist):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    # 3D Gaussian待优化的参数
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


# 函数目的：根据输入的深度图，通过阈值 config['mapping']['sil_thres'] 等一系列操作生成掩码，然后在场景中添加新的高斯分布。这些高斯分布代表了场景中的新结构。
# 传参：当前模型参数 params、变量 variables、密集化数据 densify_curr_data，以及一些配置参数，如阈值、时间索引等。
def add_new_gaussians(params, variables, curr_data, sil_thres, time_idx, mean_sq_dist_method):
    # Silhouette Rendering
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts)
    # 通过渲染器 Renderer 得到深度图和轮廓图，其中 depth_sil 包含了深度信息和轮廓信息，silhouette取出轮廓信息
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    non_presence_sil_mask = (silhouette < sil_thres) # S(p) < 0.5
    # Check for new foreground objects by using GT depth
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    # A.致密化Mask M(p)的创建：对应论文的公式(9)
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    # B.对于每个像素，基于此掩模，生成新的高斯分布参数，并将这些参数添加到原有的高斯分布参数中
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        # 重点函数：get_pointcloud()与initialize_new_params()
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist)
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    return params, variables
