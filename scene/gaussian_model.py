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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, quaternion_to_matrix
from scene.neural_renderer import GridRenderer

class GaussianModel:


    def proximity(self, scene_extent, N = 3):
        dist, nearest_indices = distCUDA2(self.get_xyz)
        selected_pts_mask = torch.logical_and(dist > (5. * scene_extent),
                                              torch.max(self.get_scaling, dim=1).values > (scene_extent))

        new_indices = nearest_indices[selected_pts_mask].reshape(-1).long()
        source_xyz = self._xyz[selected_pts_mask].repeat(1, N, 1).reshape(-1, 3)
        target_xyz = self._xyz[new_indices]
        new_xyz = (source_xyz + target_xyz) / 2
        new_scaling = self._scaling[new_indices]
        new_rotation = torch.zeros_like(self._rotation[new_indices])
        new_rotation[:, 0] = 1
        new_features_dc = torch.zeros_like(self._features_dc[new_indices])
        new_features_rest = torch.zeros_like(self._features_rest[new_indices])
        new_opacity = self._opacity[new_indices]
        new_features = torch.zeros_like(self._features[new_indices])
        new_depth_err = torch.zeros_like(self._depth_err[new_indices])
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_features, new_opacity, new_scaling, new_rotation, new_depth_err)

    def random_del_and_split_gaussians(self, del_mask, del_ratio, split_mask, N):
        ratio_mask = torch.bernoulli(torch.ones_like(del_mask, dtype=torch.float32)*del_ratio).bool()
        del_mask = torch.logical_and(del_mask, ratio_mask)

        stds = self.get_scaling[split_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[split_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[split_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[split_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[split_mask].repeat(N,1)
        new_features_dc = self._features_dc[split_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[split_mask].repeat(N,1,1)
        new_features = self._features[split_mask].repeat(N,1)
        new_opacity = self._opacity[split_mask].repeat(N,1)
        new_depth_err = self._depth_err[split_mask].repeat(N,1)

        # del_mask = torch.logical_or(del_mask, split_mask)

        self.prune_points(del_mask)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_features, new_opacity, new_scaling, new_rotation, new_depth_err)

    def depth_update_xyz(self, xyz, mask):
        # 根据mask和xyz更新self._xyz和优化器参数
        new_xyz = xyz[mask]
        new_features_dc = self._features_dc[mask]
        # new_features_dc[..., [0, 1, 2]] = torch.tensor([10., 0., 0.]).cuda()   ####ding for vis
        new_features_rest = self._features_rest[mask]
        new_features = self._features[mask]
        new_opacities = self._opacity[mask]
        new_scaling = self._scaling[mask]
        new_rotation = self._rotation[mask]
        new_depth_err = self._depth_err[mask]

        # self._features_dc[~mask] = torch.tensor([0., 10., 0.]).cuda()   ####ding for vis
        # self._opacity[...] *= 0.5 ####ding for vis
        self.prune_points(mask)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_features, new_opacities, new_scaling, new_rotation, new_depth_err)

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, dataset : str):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self._depth_err = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.opa_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.denom_2 = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.neural_renderer = None
        self.dataset = dataset
        self.setup_functions()

    def capture(self):
        if self.dataset == "DTU":
            return (
                self.active_sh_degree,
                self._xyz,
                # self._features,
                # self._features_dc,
                # self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.opa_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
                self.neural_renderer.state_dict(),
            )
        elif self.dataset == "LLFF":
            return (
                self.active_sh_degree,
                self._xyz,
                # self._features,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.opa_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
                self.neural_renderer.state_dict(),
            )
        else:
            return (
                self.active_sh_degree,
                self._xyz,
                # self._features,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.opa_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
                self.neural_renderer.state_dict(),
            )
    
    def restore(self, model_args, training_args=None):
        if self.dataset == "DTU":
            (self.active_sh_degree, 
            self._xyz, 
            # self._features,
            # self._features_dc, 
            # self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum,
            opa_gradient_accum,
            denom,
            opt_dict, 
            self.spatial_lr_scale,
            neural_renderer_state) = model_args
        elif self.dataset == "LLFF":
            (self.active_sh_degree, 
            self._xyz, 
            # self._features,
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum,
            opa_gradient_accum,
            denom,
            opt_dict, 
            self.spatial_lr_scale,
            neural_renderer_state) = model_args
        else:
            (self.active_sh_degree, 
            self._xyz, 
            # self._features,
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum,
            opa_gradient_accum,
            denom,
            opt_dict, 
            self.spatial_lr_scale,
            neural_renderer_state) = model_args
        if training_args is not None:
            self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.opa_gradient_accum = opa_gradient_accum
        self.denom = denom
        if self.optimizer is not None:
            self.optimizer.load_state_dict(opt_dict)

        if self.optimizer is None:
            self.neural_renderer = GridRenderer()
        self.neural_renderer.recover_from_ckpt(neural_renderer_state)
        self.neural_renderer.cuda()

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):

        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
        # return self._features
        # raise NotImplementedError

    @property
    def get_dino_features(self):
        return self._features
    
    @property
    def get_opacity(self):
        # return self.opacity_activation(self._opacity)
        sigma = self.neural_renderer.density(self.get_xyz)['sigma']
        _opa_neu = sigma.view(-1,1)
        return self.combine_opacity(_opa_neu)
    
    @property
    def get_opacity_(self):
        return self.opacity_activation(self._opacity)
        
    def combine_opacity(self, opa_neu):
        # opa_pnt = self.opacity_activation(self._opacity + opa_neu)
        opa_pnt = self.opacity_activation(self._opacity)
        # opa_pnt = self.opacity_activation(opa_neu)
        # return torch.maximum(opa_neu, opa_pnt)
        return opa_pnt
    
    @property
    def get_depth_err(self):
        return torch.nn.functional.softplus(self._depth_err)
        
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # print(f'max_sh_degree:{self.max_sh_degree}')
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        # features = torch.zeros((fused_point_cloud.shape[0], 16)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda())[0], 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._features = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 32)).float().cuda().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._depth_err = nn.Parameter(torch.zeros_like(opacities).requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.neural_renderer = GridRenderer(
                                    bound=((self._xyz.max(0).values-self._xyz.min(0).values).max())/2 * 1.2,
                                    coord_center=self._xyz.mean(0)
                                ).cuda()
    

    def create_from_pcd_with_opa(self, pcd : BasicPointCloud, spatial_lr_scale : float, opa):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        # features = torch.zeros((fused_point_cloud.shape[0], 16)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda())[0], 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        assert(opa.shape == (fused_point_cloud.shape[0], 1))
        opacities = inverse_sigmoid(opa)
        # 0.9 * torch.ones((points.shape[0], 1), dtype=torch.float32, device='cuda')
        # opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._features = nn.Parameter(torch.zeros((fused_point_cloud.shape[0], 32)).float().cuda().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._depth_err = nn.Parameter(torch.zeros_like(opacities).requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.neural_renderer = GridRenderer(
                                    bound=((self._xyz.max(0).values-self._xyz.min(0).values).max())/2 * 1.2,
                                    coord_center=self._xyz.mean(0)
                                ).cuda()

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.opa_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_2 = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._features], 'lr': 0.001, "name": "features"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._depth_err], 'lr': training_args.scaling_lr, "name": "depth_err"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        l += self.neural_renderer.get_params(lr=training_args.neural_grid, lr_net=training_args.neural_net)

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_steps=training_args.position_lr_delay_steps,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        if self.dataset == "LLFF" or self.dataset == "blender":
            # All channels except the 3 DC
            for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
            # for i in range(self._features.shape[1]):
            #     l.append('features_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        if self.dataset == "LLFF" or self.dataset == "blender":
            f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self._opacity.detach().cpu().numpy()
            # features = self._features.detach().cpu().numpy()
        opacities = self.inverse_opacity_activation(self.get_opacity.detach()).cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if self.dataset == "LLFF" or self.dataset == "blender":
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elif self.dataset == "DTU":
            attributes = np.concatenate((xyz, normals, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_ply_color(self, path, color):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # opacities = self._opacity.detach().cpu().numpy()
        # features = self._features.detach().cpu().numpy()
        opacities = self.inverse_opacity_activation(self.get_opacity.detach()).cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        color = color.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()] + [(ch, "u1") for ch in ['red', 'green', 'blue']]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, opacities, scale, rotation, color * 255), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.05))
        if len(self.optimizer.state.keys()):
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]


    def reset_opacity_max(self, value=0.8):
        opacities_new = inverse_sigmoid(torch.max(self.get_opacity, torch.ones_like(self.get_opacity)*value))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        # raise NotImplementedError

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]


        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # self._features = nn.Parameter(torch.tensor(features, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is None: continue
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'neural' in group["name"]: continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._features = optimizable_tensors["features"]
        self._opacity = optimizable_tensors["opacity"]
        self._depth_err = optimizable_tensors["depth_err"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.opa_gradient_accum = self.opa_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.denom_2 = self.denom_2[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def prune_points_inference(self, mask):
        valid_points_mask = ~mask

        self._xyz = self._xyz[valid_points_mask.expand(self._xyz.shape)].view(-1,3)
        self._opacity = self._opacity[valid_points_mask.expand(self._opacity.shape)].view(-1,1)
        self._scaling = self._scaling[valid_points_mask.expand(self._scaling.shape)].view(-1,3)
        self._rotation = self._rotation[valid_points_mask.expand(self._rotation.shape)].view(-1,4)


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'neural' in group["name"]: continue
            assert len(group["params"]) == 1

            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, 
                              new_features_rest, new_features, 
                              new_opacities, new_scaling, new_rotation, 
                              new_depth_err):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "features": new_features,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "depth_err" : new_depth_err}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._features = optimizable_tensors["features"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._depth_err = optimizable_tensors["depth_err"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.opa_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_2 = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, opa_thresh, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask_base = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask_base = torch.logical_and(selected_pts_mask_base,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if self.dataset == "DTU":
            selected_pts_mask = torch.logical_and(selected_pts_mask_base,
                                                self.get_opacity.squeeze() > opa_thresh)
        if self.dataset == "LLFF" or self.dataset == "blender":
            # selected_pts_mask = torch.logical_or(selected_pts_mask,
            #                                       selected_pts_mask_base * (self.get_scaling.max(dim=1).values / self.get_scaling.min(dim=1).values > 10))
            
            dist, _ = distCUDA2(self.get_xyz)
            selected_pts_mask2 = torch.logical_and(dist > (10. * scene_extent),
                                                torch.max(self.get_scaling, dim=1).values > ( scene_extent))
            selected_pts_mask = torch.logical_or(selected_pts_mask_base, selected_pts_mask2)


        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        # samples = torch.clamp(samples, max=torch.ones_like(samples) * scene_extent * 0.1, min=-torch.ones_like(samples) * scene_extent * 0.1)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_features = self._features[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_depth_err = self._depth_err[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_features, new_opacity, new_scaling, new_rotation, new_depth_err)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, opa_thresh):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if self.dataset == "DTU":
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                self.get_opacity.squeeze() > opa_thresh)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_features = self._features[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_depth_err = self._depth_err[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        
            
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_features, new_opacities, new_scaling, new_rotation, new_depth_err)



    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, opa_thresh, max_dist):
        if max_dist is not None:
            outliers_mask = torch.sqrt(distCUDA2(self._xyz.detach()).view(max_dist.shape)[0]) > max_dist
            self.prune_points(outliers_mask.squeeze())
        
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, opa_thresh)
        self.densify_and_split(grads, max_grad, extent, opa_thresh)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask += big_points_vs

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, accum_denom=True, add_random=False):
        # increase randomness to overcome local minima
        # print(f"grad shape: {viewspace_point_tensor.grad.shape}, {update_filter.shape}")
        if add_random:
            self.xyz_gradient_accum[update_filter] += torch.norm((viewspace_point_tensor.grad - self._xyz.grad)[update_filter,:2], dim=-1, keepdim=True)
        else:
            self.xyz_gradient_accum[update_filter] += torch.norm((viewspace_point_tensor.grad)[update_filter,:2], dim=-1, keepdim=True)
        if accum_denom:
            self.denom[update_filter] += 1


    def prune(self, min_opacity, max_err=None):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()



