from utils.camera_pose import project_to_image, quaternion_to_rotation_matrix, world_to_camera
from utils.keyframe_union_mask import create_point_mask


def stm_mask(params,dataset,time_idx, depth):          
    cam_rot = quaternion_to_rotation_matrix(params['cam_unnorm_rots'][..., time_idx]).to('cpu')
    cam_tran = params['cam_trans'][..., time_idx].to('cpu')
    camer = [dataset.orig_width,dataset.orig_height,dataset.fx,dataset.fy,dataset.cx,dataset.cy]
    camera_points = world_to_camera(params['means3D'], cam_rot, cam_tran)
    image_points = project_to_image(camera_points, camer[2], camer[3], camer[4], camer[5])
    # mask1 = (image_points[:, 0] >= camer[0]*0.1) & (image_points[:, 0] < camer[0]*0.9) & \
    #     (image_points[:, 1] >= camer[1]*0.1) & (image_points[:, 1] < camer[1]*0.9)
    #     # (camera_points[:, 2] >=0) & (camera_points[:, 2] < (depth.mean()+1*torch.std(depth)))
    
    mask = create_point_mask(image_points, depth)
    return mask