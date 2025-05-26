from utils.camera_pose import project_to_image, quaternion_to_rotation_matrix, world_to_camera

def racall_mask(params,time_idx,dataset,proportion_memory):
    M =proportion_memory
    cam_rot = quaternion_to_rotation_matrix(params['cam_unnorm_rots'][..., time_idx]).to('cpu')
    cam_tran = params['cam_trans'][..., time_idx].to('cpu')
    camer = [dataset.orig_width,dataset.orig_height,dataset.fx,dataset.fy,dataset.cx,dataset.cy]
    camera_points = world_to_camera(params['means3D'], cam_rot, cam_tran)
    image_points = project_to_image(camera_points, camer[2], camer[3], camer[4], camer[5])
    mask = (image_points[:, 0] >= -camer[0]*M) & (image_points[:, 0] < camer[0]*(1+M)) & \
        (image_points[:, 1] >= -camer[1]*M) & (image_points[:, 1] < camer[1]*1+M)
    return mask