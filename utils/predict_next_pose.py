
import torch
from torch.linalg import inv
from scipy.linalg import expm

def quaternion_to_rotation_matrix_c(q: torch.Tensor) -> torch.Tensor:

    w, x, y, z = q[...,0], q[...,1], q[...,2], q[...,3]
    return torch.stack([
        1-2*(y**2+z**2),   2*(x*y-z*w),   2*(x*z+y*w),
        2*(x*y+z*w),   1-2*(x**2+z**2),   2*(y*z-x*w),
        2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x**2+y**2)
    ], dim=-1).view(*q.shape[:-1], 3, 3)

def matrix_exp(w: torch.Tensor) -> torch.Tensor:

    if w.dim() == 1:  # 旋转向量
        theta = torch.norm(w)
        if theta < 1e-6:
            return torch.eye(3, device=w.device)
        w_skew = torch.tensor([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ], device=w.device) / theta
        return torch.eye(3, device=w.device) + \
               torch.sin(theta) * w_skew + \
               (1 - torch.cos(theta)) * (w_skew @ w_skew)
    elif w.dim() == 2:  # 反对称矩阵
        theta = torch.norm(w)
        if theta < 1e-6:
            return torch.eye(3, device=w.device)
        return torch.eye(3, device=w.device) + \
               torch.sin(theta) * w + \
               (1 - torch.cos(theta)) * (w @ w)
    else:
        raise ValueError("Input must be a 3D vector or 3x3 skew-symmetric matrix")

def optimize_trajectory(params: dict, time_idx: int, window_size=5):
    """
    滑动窗口位姿图优化（建议每5-10帧调用一次）
    优化策略：最小化相邻位姿的相对运动残差
    """
    device = params['cam_trans'].device
    start = max(0, time_idx - window_size + 1)
    end = time_idx + 1
    
    # 提取窗口数据
    trans = params['cam_trans'][0, :, start:end]  # [3, N]
    rots = params['cam_unnorm_rots'][0, :, start:end]  # [4, N]
    
    # 转换为位姿矩阵 [N, 4, 4]
    R = quaternion_to_rotation_matrix_c(rots.permute(1,0))  # [N, 3, 3]
    T = torch.eye(4, device=device).repeat(R.shape[0],1,1)
    T[:, :3, :3] = R
    T[:, :3, 3] = trans.T
    
    # 构建最小二乘问题：最小化相邻位姿变换差异
    for _ in range(2):  # 迭代2次足够实时性要求
        for i in range(1, T.shape[0]):
            # 计算相对变换残差
            delta = inv(T[i-1]) @ T[i]
            
            # 平移残差加权优化（权重与时间间隔相关）
            trans_weight = 1.0 / (torch.norm(T[i,:3,3]-T[i-1,:3,3]) + 1e-6)
            T[i,:3,3] -= 0.5 * trans_weight * delta[:3,3]
            
            # 旋转残差采用李代数更新
            theta = torch.acos((torch.trace(delta[:3,:3])-1)/2)
            if theta > 1e-6:
                lnR = theta/(2*torch.sin(theta)) * (delta[:3,:3] - delta[:3,:3].T)
                w = torch.stack([lnR[2,1], lnR[0,2], lnR[1,0]])
                T[i,:3,:3] = T[i,:3,:3] @ matrix_exp(-0.3*w)
    
    # 更新参数（仅更新中间帧，保持首尾约束）
    mid_start = max(1, start+1)
    mid_end = min(end-1, time_idx)
    params['cam_trans'][0, :, mid_start:mid_end] = T[1:-1, :3, 3].T
    params['cam_unnorm_rots'][0, :, mid_start:mid_end] = \
        torch.tensor(R[1:-1], device=device).reshape(-1,4).T  # 需实现rotation_matrix_to_quaternion
    
    return params

def predict_next_pose(params: dict, time_idx: int):

    # 确保有足够的历史数据
    min_history = max(4, time_idx) if time_idx < 4 else 4

    # 平移预测使用4帧数据进行三阶差分
    t = params['cam_trans'][0, :, max(0, time_idx-3):time_idx+1]  # [3, 4]
    
    # 旋转预测使用3帧数据
    r = params['cam_unnorm_rots'][0, :, max(0, time_idx-2):time_idx+1]  # [4, 3]
    
    #################### 平移预测改进 ####################
    if t.shape[-1] >= 4:
        # 计算三阶差分（加加速度模型）
        v = [t[...,i] - t[...,i-1] for i in range(1,4)]  # 速度序列
        a = [v[i] - v[i-1] for i in range(1,3)]          # 加速度序列
        j = a[1] - a[0]                                  # 加加速度
        
        # 使用匀加加速运动学方程
        pred_trans = t[...,-1] + v[-1] + 0.5*a[-1] + (1/6)*j
    else:
        # 历史数据不足时使用改进的二阶差分
        delta1 = t[...,-2] - t[...,-3]
        delta2 = t[...,-1] - t[...,-2]
        accel = delta2 - delta1
        pred_trans = t[...,-1] + delta2 + 0.5*accel

    #################### 旋转预测改进 ####################
    # 将四元数转换为旋转向量进行预测
    def quat_to_rotvec(q):
        angle = 2 * torch.acos(q[0])
        axis = q[1:] / torch.norm(q[1:])
        return axis * angle
    
    def rotvec_to_quat(rv):
        angle = torch.norm(rv)
        axis = rv / angle if angle > 1e-6 else torch.zeros_like(rv)
        return torch.cat([torch.cos(angle/2).unsqueeze(0), axis*torch.sin(angle/2)])
    
    # 计算历史旋转向量
    rvecs = torch.stack([quat_to_rotvec(r[:,i]) for i in range(r.shape[1])], dim=1)
    
    # 计算角速度和角加速度
    omega1 = rvecs[:,1] - rvecs[:,0]
    omega2 = rvecs[:,2] - rvecs[:,1]
    alpha = omega2 - omega1
    
    # 预测下一旋转向量
    pred_rvec = rvecs[:,2] + omega2 + 0.5*alpha
    pred_quat = rotvec_to_quat(pred_rvec)
    pred_quat = pred_quat / torch.norm(pred_quat)

    #################### 参数更新 ####################
    params['cam_trans'][0, :, time_idx+1] = pred_trans
    params['cam_unnorm_rots'][0, :, time_idx+1] = pred_quat
    
    return params

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:

    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=-1)

def quaternion_inverse(q: torch.Tensor) -> torch.Tensor:

    return torch.stack([q[0], -q[1], -q[2], -q[3]])
