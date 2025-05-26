import os
import cv2
from matplotlib import pyplot as plt
import torch
from utils.slam_external import calc_ssim
from utils.slam_helpers import transformed_params2rendervar, l1_loss_v1, transform_to_frame, transformed_params2depthplussilhouette
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1,ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None):
    losses = {}

    if tracking:
        transformed_pts = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    
    elif mapping:
        if do_ba:
            transformed_pts = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        
        
        else:
            transformed_pts = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        transformed_pts = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_pts)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts)

    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D'] 

   
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)

    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)

    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()



    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))

    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)

    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
 
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask


    if use_l1:
        mask = mask.detach()
        if tracking:
            
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:

            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    


    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking:

        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:

        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))



    # Visualize the Diff Images
    if tracking and visualize_tracking_loss:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Loss Mask")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
        plt.close()
        plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
        cv2.imshow('Diff Images', plot_img)
        cv2.waitKey(1)
        ## Save Tracking Loss Viz
        # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
        # os.makedirs(save_plot_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
        # plt.close()

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}

    loss = sum(weighted_losses.values())

    if mapping:
        seen = radius > 0  
        variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
        variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses