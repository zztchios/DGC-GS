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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, K, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, depth_mono, npy_depth, depth_npy_nomask,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.K = K
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name


        """depth_scale"""
        self.depth_scale = None
        self.acc_depth_gts = []
        self.acc_depth_renders = []
        """depth_scale"""

        """grad_depth"""
        self.grad_gt = None
        """grad_depth"""

        """grad_split_del"""
        self.acc_depth_grad = None
        """grad_split_del"""

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.data_device)

        """npy_depth"""
        self.npy_depth = None
        if npy_depth is not None:
            self.npy_depth = npy_depth.to(self.data_device)
        """npy_depth"""

        """npy_depth_nomask"""
        self.depth_npy_nomask = None
        if depth_npy_nomask is not None:
            self.depth_npy_nomask = depth_npy_nomask.to(self.data_device)
        """npy_depth_nomask"""

        self.depth_mono = None
        self.original_image = None
        if depth_mono is not None:
            self.depth_mono = depth_mono.to(self.data_device)
        # self.mono_scale = torch.nn.parameter.Parameter(data=torch.tensor(1.0, device=data_device), requires_grad=True)
        # self.mono_bias = torch.nn.parameter.Parameter(data=torch.tensor(0.0, device=data_device), requires_grad=True)
        # self.mono_optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        if image is not None:
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]

            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class PseudoCamera(nn.Module):
    def __init__(self, R, T, FoVx, FoVy,
                 uid, cam_uid, 
                 image,
                 image_name,
                 gt_alpha_mask,
                 width, height, 
                 trans=np.array([0.0, 0.0, 0.0]), 
                 scale=1.0, data_device = "cuda"):
        super(PseudoCamera, self).__init__()

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.uid = uid
        self.cam_uid = cam_uid
        self.image_width = width
        self.image_height = height
        self.image_name = image_name
        self.grad_gt = None
        self.warp_mask = None
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        if torch.cuda.is_available():
            torch.cuda.set_device(self.data_device)
            
        # if depth_mono is not None:
        #     self.depth_mono = depth_mono.to(self.data_device)
        self.depth_mono = None
        if image is not None:
            self.org_image = image.to(self.data_device)
        else:
            self.org_image = None

        self.original_image = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def set_warp_image(self, image):
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        
    def set_warp_depth(self, depth):
        self.depth_mono = depth.to(self.data_device)

    def set_warp_mask(self, mask):
        self.warp_mask = mask.to(self.data_device)


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

