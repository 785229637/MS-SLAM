import torch
import torch.nn.functional as F

def initialize_camera_pose(params, curr_time_idx, forward_prop):
    with torch.no_grad(): 
        if curr_time_idx > 1 and forward_prop: 

            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()

            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else:

            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
    
    return params


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store




def world_to_camera(points, cam_rot, cam_tran):

    if cam_rot.shape == (3, 3):
        identity = torch.eye(4)
        identity[:3, :3] = cam_rot
        cam_rot = identity
    
    cam_tran = cam_tran.unsqueeze(0)  
    cam_tran_homogeneous = torch.cat([cam_tran[0].to("cuda:0"), torch.ones(1, 1).to("cuda:0")], dim=1)  # 变为 (1, 4)

    points_homogeneous = torch.cat([points, torch.ones(points.shape[0], 1).to("cuda:0")], dim=1)  # (N, 4)

    camera_points = cam_rot.to("cuda:0") @ points_homogeneous.T  # (4, N)
    camera_points = camera_points.T + cam_tran_homogeneous.to("cuda:0")  # (N, 4)

    return camera_points[:, :3]  




def quaternion_to_rotation_matrix(quaternion):
    q = quaternion[0]
    q_matrix = torch.tensor([
        [1 - 2 * q[2]**2 - 2 * q[3]**2, 2 * q[1] * q[2] - 2 * q[3] * q[0], 2 * q[1] * q[3] + 2 * q[2] * q[0],0],
        [2 * q[1] * q[2] + 2 * q[3] * q[0], 1 - 2 * q[1]**2 - 2 * q[3]**2, 2 * q[2] * q[3] - 2 * q[1] * q[0],0],
        [2 * q[1] * q[3] - 2 * q[2] * q[0], 2 * q[2] * q[3] + 2 * q[1] * q[0], 1 - 2 * q[1]**2 - 2 * q[2]**2,0],
        [0, 0, 0, 1]
    ], dtype=torch.float32)
    return q_matrix
def project_to_image(camera_points, fx, fy, cx, cy):

    x = camera_points[:, 0] / camera_points[:, 2] * fx + cx
    y = camera_points[:, 1] / camera_points[:, 2] * fy + cy
    return torch.stack([x, y], dim=1)

