
import time

import numpy as np
import torch
from scripts.loss import get_loss
from utils.eval_helpers import report_loss, report_progress
from utils.slam_external import densify, prune_gaussians


def msmap(params_map,variables_map,num_iters_mapping,progress_bar,selected_keyframes,time_idx,gt_w2c_all_frames,intrinsics,first_frame_w2c,cam,config,optimizer,keyframe_list,color,depth):
    
    for iter in range(num_iters_mapping):
        iter_start_time = time.time()
        # Randomly select a frame until current time step amongst keyframes
        rand_idx = np.random.randint(0, len(selected_keyframes))
        selected_rand_keyframe_idx = selected_keyframes[rand_idx]
        if selected_rand_keyframe_idx == -1:
            # Use Current Frame Data
            iter_time_idx = time_idx
            iter_color = color
            iter_depth = depth
        else:
            # Use Keyframe Data
            iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
            iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
            iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
        iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
        iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                        'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
        # Loss for current frame

        loss, variables_map, losses = get_loss(params_map, iter_data, variables_map, iter_time_idx, config['mapping']['loss_weights'],
                                        config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                        config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True,do_ba=True)
        # Backprop
        loss.backward()
        with torch.no_grad():

            if config['mapping']['prune_gaussians']:
                params_map, variables_map = prune_gaussians(params_map, variables_map, optimizer, iter, config['mapping']['pruning_dict'])

            if config['mapping']['use_gaussian_splatting_densification']:
                params_map, variables_map = densify(params_map, variables_map, optimizer, iter, config['mapping']['densify_dict'])

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # Report Progress
            if config['report_iter_progress']:
                report_progress(variables_map, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                    mapping=True, online_time_idx=time_idx)
            else:
                progress_bar.update(1)
        # Update the runtime numbers
        iter_end_time = time.time()
        return(params_map,variables_map,iter_end_time,iter_start_time)