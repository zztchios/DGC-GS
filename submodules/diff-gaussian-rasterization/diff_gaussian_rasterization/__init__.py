#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C


def select_tiles(
        means3D,
        opacities,
        tiles,
        K,h,w,
        alpha_thresh,
        scales, 
        rotations,
        cov3Ds_precomp,
        raster_settings, ):
    args = (
        means3D,
        opacities,
        tiles,
        K,h,w,
        alpha_thresh,
        scales,
        rotations,
        raster_settings.scale_modifier,
        cov3Ds_precomp,
        raster_settings.viewmatrix,
        raster_settings.projmatrix,
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        raster_settings.image_height,
        raster_settings.image_width,
        raster_settings.campos,
        raster_settings.prefiltered,
        raster_settings.debug
    )

    # Invoke C++/CUDA rasterizer
    if raster_settings.debug:
        cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
        try:
            idxs, alphas = _C.select_tiles(*args)
        except Exception as ex:
            torch.save(cpu_args, "snapshot_sl.dump")
            print("\nAn error occured in select. Please forward snapshot_sl.dump for debugging.")
            raise ex
    else:
        idxs, alphas = _C.select_tiles(*args)
    ### idxs\alphas: (tiles.shape[0], K)
    return idxs, alphas


def select_gaussians(
    means3D,
    opacities,
    gt_depth,
    depth_opa_thresh,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    args = (
        means3D,
        opacities,
        gt_depth,
        depth_opa_thresh,
        scales,
        rotations,
        raster_settings.scale_modifier,
        cov3Ds_precomp,
        raster_settings.viewmatrix,
        raster_settings.projmatrix,
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        raster_settings.image_height,
        raster_settings.image_width,
        raster_settings.campos,
        raster_settings.prefiltered,
        raster_settings.debug
    )

    # Invoke C++/CUDA rasterizer
    
    if raster_settings.debug:   # False
        cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
        try:
            change_idx, change_xyz = _C.select_gaussians(*args)
        except Exception as ex:
            torch.save(cpu_args, "snapshot_sl.dump")
            print("\nAn error occured in select. Please forward snapshot_sl.dump for debugging.")
            raise ex
    else:
        change_idx, change_xyz = _C.select_gaussians(*args)
    
    return change_idx, change_xyz

def precompute_depth_scale(
    means3D,
    raster_settings,
):
    args = (
        means3D,
        raster_settings.viewmatrix,
        raster_settings.projmatrix,
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        raster_settings.image_height,
        raster_settings.image_width,
        raster_settings.prefiltered,
        raster_settings.debug
    )

    # Invoke C++/CUDA rasterizer
    if raster_settings.debug:
        cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
        try:
            pix_mask, pix_depth = _C.precompute_depth_scale(*args)
        except Exception as ex:
            torch.save(cpu_args, "snapshot_sl.dump")
            print("\nAn error occured in select. Please forward snapshot_sl.dump for debugging.")
            raise ex
    else:
        pix_mask, pix_depth = _C.precompute_depth_scale(*args)  # 仿照这个写的_C.rasterize_gaussians
    
    return pix_mask, pix_depth


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, alpha, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, alpha, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha)
        return color, radii, depth, alpha

    @staticmethod
    def backward(ctx, grad_color, grad_radii, grad_depth, grad_alpha):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_color,
                grad_depth,
                grad_alpha,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                alpha,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads
    

def alpha_select(my_ctx, pixels, thresh=0.1):

    # num_rendered = ctx.num_rendered
    # raster_settings = ctx.raster_settings
    # colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, alpha = ctx.saved_tensors
    pix_count = pixels.shape[0]
    # P = means3D.shape[0]
    raster_settings, num_rendered, P, geomBuffer, binningBuffer, imgBuffer = my_ctx
    args = (pixels,
            pix_count, 
            raster_settings.image_height,
            raster_settings.image_width,
            P,
            num_rendered, thresh, 
            geomBuffer,
            binningBuffer, 
            imgBuffer, 
            raster_settings.debug)

    # Compute gradients for relevant tensors by invoking backward method
    if raster_settings.debug:
        cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
        try:
            ranges, idxs = _C.alphaSelect(*args)
        except Exception as ex:
            torch.save(cpu_args, "snapshot_bw.dump")
            print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
            raise ex
    else:
            ranges, idxs = _C.alphaSelect(*args)
        
    return ranges, idxs



class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )
    
    def select_depth_gaussian(self, means3D, opacities, gt_depth, scales = None, rotations = None, cov3D_precomp = None, depth_opa_thresh = 0.1):
        
        raster_settings = self.raster_settings
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return select_gaussians(
            means3D,
            opacities,
            gt_depth,
            depth_opa_thresh,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )
    
    def precompute_depth_scale(self, means3D):
        
        raster_settings = self.raster_settings

        # Invoke C++/CUDA rasterization routine
        return precompute_depth_scale(
            means3D,
            raster_settings, 
        )
    
    def alpha_select_tiles(self, means3D, opacities, tiles, K,h,w, scales = None, rotations = None, cov3D_precomp = None, alpha_thresh = 0.1):
        
        raster_settings = self.raster_settings
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return select_tiles(
            means3D,
            opacities,
            tiles,
            K,h,w,
            alpha_thresh,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )




