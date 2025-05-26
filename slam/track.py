import torch
from tqdm import tqdm

import time

from scripts.loss import get_loss
from scripts.optimizer import initialize_optimizer
from utils.eval_helpers import report_loss, report_progress

def track(params_tric, variables,config,time_idx,tracking_curr_data, iter_time_idx ,eval_dir):
    optimizer = initialize_optimizer(params_tric, config['tracking']['lrs'], tracking=True)

    candidate_cam_unnorm_rot = params_tric['cam_unnorm_rots'][..., time_idx].detach().clone()
    candidate_cam_tran = params_tric['cam_trans'][..., time_idx].detach().clone()
    current_min_loss = float(1e20)
    
    # Tracking Optimization
    iter = 0
    do_continue_slam = False
    num_iters_tracking = config['tracking']['num_iters']
    progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}") 
    

    while True:
        iter_start_time = time.time() # 计算迭代开始的时间
        # Loss for current frame
        # 重点函数：计算当前帧的损失
        loss, variables, losses = get_loss(params_tric, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                            config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                            config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, 
                                            plot_dir=eval_dir, visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                            tracking_iteration=iter)

        
        # Backprop 将loss进行反向传播,计算梯度
        loss.backward()
        
        # Optimizer Update
        # 更新模型参数
        optimizer.step()
        # 清除已计算的梯度
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            # Save the best candidate rotation & translation  
            # 如果当前损失小于 current_min_loss，更新最小损失对应的相机旋转和平移
            if loss < current_min_loss:
                current_min_loss = loss
                candidate_cam_unnorm_rot = params_tric['cam_unnorm_rots'][..., time_idx].detach().clone()
                candidate_cam_tran = params_tric['cam_trans'][..., time_idx].detach().clone()
            
            # Report Progress
            if config['report_iter_progress']:

                report_progress(params_tric, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
            else:
                progress_bar.update(1)
        
        # Update the runtime numbers 更新迭代次数和计算迭代的运行时间
        iter_end_time = time.time()
        tracking_iter_time = iter_end_time - iter_start_time
        

        # Check if we should stop tracking 检查是否最大迭代次数，满足终止计算
        iter += 1
        if iter == num_iters_tracking:
            if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                break
            elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                do_continue_slam = True
                progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                num_iters_tracking = 2*num_iters_tracking
            else:
                break

    # ** Sec 1.4 数据更新与进度跟踪 **
    # 这里从while循环出来了,更新最佳候选
    progress_bar.close()
    return candidate_cam_unnorm_rot , candidate_cam_tran,tracking_iter_time
    # Copy over the best candidate rotation & translation