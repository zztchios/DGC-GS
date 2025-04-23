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

import os
import random
import json
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_sh import GaussianModelSH
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, renderCameraList_from_camInfos

""""""
from utils.depth_utils import generate_ply_from_depth
from scene.gaussian_model import BasicPointCloud

from utils.pose_utils import generate_alphas_poses_llff, generate_alphas_poses_dtu, generate_alphas_poses_360, generate_alphas_poses_blender
from utils.pose_utils import generate_interpolated_poses
from scene.cameras import PseudoCamera
import numpy as np
from PIL import Image

""""""

def img_to_PIL(img, error=False):
    img = (img - img.min()) * (255 / (img.max() - img.min()))
    img = img.detach().cpu().numpy()
    
    if (len(img.shape) == 2) or (img.shape[0] == 1):
        img = np.squeeze(img)
        if error:           
            # 转换回PIL Image
            img = Image.fromarray(img.astype('uint8'))
        else:
            img = Image.fromarray(img.astype('uint8'))
    else:
        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray(img.astype('uint8'))
    return img

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], init_with_gt_depth=False, slerp_open=False, pseudo_views=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.source_path = args.source_path 
        self.loaded_iter = None
        self.gaussians = gaussians
        # print(f'load_iteration:{load_iteration}') # load_iteration:None
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras  = {}
        self.test_cameras   = {}
        self.eval_cameras   = {}
        #--------------------zzt--------------------
        self.target_cameras = {}
        self.slerp_cameras  = {}
        self.intrinsic      = {}
        #--------------------zzt--------------------

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.dataset, args.eval, args.rand_pcd, args.mvs_pcd, N_sparse = args.n_sparse)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.rand_pcd, N_sparse = args.n_sparse)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            if scene_info.eval_cameras:
                camlist.extend(scene_info.eval_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.eval_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras", resolution_scales)
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras", resolution_scales)
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print("Loading Eval Cameras", resolution_scales)
            self.eval_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.eval_cameras, resolution_scale, args)

            #--------------------zzt--------------------
            if not args.render:
                # 计算内参
                cam_focal_dict = dict()
                '''
                {'id': 0, 'img_name': 'DJI_20200223_163016_842', 'width': 4032, 
                'height': 3024, 'position': [-4.843687316974395, -2.961359711313386, -1.7286079149831672], 
                'rotation': [[0.9915760170988446, -0.023984008701182398, 0.12728617223018918], [0.012397962579199612, 0.9957694211317604, 0.09104696844378514], [-0.128931349323105, -0.08870190113825321, 0.9876784293970294]], 
                'fy': 3368.8237176028883, 'fx': 3368.823717602889},
                '''    
                for i in json_cams:
                    cam_focal_dict[i['id']] = (i['fx'],i['fy'])
                train_views = self.train_cameras[resolution_scale]
                cam_focal_list = dict()
                for cam in train_views:
                    _, H, W = cam.original_image.clone().detach().shape
                    cam_focal_list[cam.uid] = torch.from_numpy(np.array([
                        [cam_focal_dict[cam.uid][0]/args.resolution, 0, (W-1) / 2],
                        [0, cam_focal_dict[cam.uid][1]/args.resolution, (H-1) / 2],
                        [0, 0, 1.0]
                        ])).to(torch.float32).to('cuda')
                self.intrinsic[resolution_scale] = cam_focal_list 
                
                # 加入虚拟的目标相机
                tgt_poses = []
                if isinstance(args.theta_range_deg, list) and isinstance(args.theta_range_deg, list):
                    # theta_range_deg = [int(x) for x in args.theta_range_deg.split()]
                    # translate_range = [float(x) for x in args.translate_range.split()]
                    for dt in zip(args.theta_range_deg, args.translate_range):
                        # 直接输出配对的元素作为列表
                        dt_list = list(dt)

                        if args.source_path.find('dtu'):
                            # t_range = [dt_list[1] * x for x in ([a - b for a, b in zip(max_T_AB, min_T_AB)])]
                            tgt_pose = generate_alphas_poses_dtu(train_views, \
                                                                theta_range_deg=dt_list[0], \
                                                                translate_range=dt_list[1])
                        elif args.source_path.find('360'):
                            tgt_pose = generate_alphas_poses_360(train_views,  \
                                                                theta_range_deg=dt_list[0], translate_range=dt_list[1])
                        elif args.source_path.find('llff'):
                            tgt_pose = generate_alphas_poses_llff(train_views,  \
                                                                theta_range_deg=dt_list[0], translate_range=dt_list[1])
                        elif args.source_path.find('nerf_synthetic'):
                            tgt_pose = generate_alphas_poses_blender(train_views, \
                                                                    theta_range_deg=dt_list[0], translate_range=dt_list[1])
                        else:
                            raise ValueError("theta_range_deg and translate_range must be list or int")
                        tgt_poses.append(tgt_pose)           
                else:
                    raise ValueError("theta_range_deg and translate_range must be list or int")
                # print("tgt_poses", tgt_poses[0].shape) # 2 [3,4,4]
                
                tgt_cams = list()
                for d, pose in enumerate(tgt_poses):
                    for idx, view in enumerate(train_views):
                        tgt_cams.append(
                            PseudoCamera(
                            R=pose[view.uid][:3, :3], T=pose[view.uid][:3, 3], FoVx=view.FoVx, FoVy=view.FoVy, # R=pose[idx][:3, :3].T, T=pose[idx][:3, 3]
                            image_name=view.image_name, uid=view.uid+100*d,
                            cam_uid = view.uid, image=view.original_image, gt_alpha_mask=None, 
                            width=view.image_width, height=view.image_height, data_device=args.data_device
                            ))

                print("Loading Target Cameras", len(tgt_cams))
                self.target_cameras[resolution_scale] = tgt_cams 
                
                if slerp_open:
                    slerp_point_cam = self.train_cameras[resolution_scale]
                    
                    slerp_cams = list()
                    if len(pseudo_views) == 1:
                        # pass
                        # 随机取一个外参之间的视角，然后这两个视角互相监督
                        for i in range(len(slerp_point_cam)):
                            for j in range(i + 1, len(slerp_point_cam)):
                                pseudo_poses = generate_interpolated_poses([slerp_point_cam[i],
                                                            slerp_point_cam[j]], 1) # [2, 4, 4]
                                slerp_and_warp_cams = list()
                                for k in range(len(pseudo_poses)):
                                    pseudo_cams = list()
                                    pseudo_view = slerp_point_cam[i]
                                    pseudo_cams.append(PseudoCamera(
                                        R=pseudo_poses[k][:3, :3], T=pseudo_poses[k][:3, 3],
                                        FoVx=pseudo_view.FoVx, FoVy=pseudo_view.FoVy,
                                        image_name=pseudo_view.image_name, uid=pseudo_view.uid+100*i,
                                        cam_uid=pseudo_view.uid, image=None, gt_alpha_mask=None, 
                                        width=pseudo_view.image_width, height=pseudo_view.image_height, 
                                        data_device=args.data_device
                                    ))    # 存储插值视角

                                    # 旋转1°
                                    if args.source_path.find('dtu'):
                                        # t_range = [dt_list[1] * x for x in ([a - b for a, b in zip(max_T_AB, min_T_AB)])]
                                        tgt_pose = generate_alphas_poses_dtu([pseudo_cams[-1]], \
                                                                            theta_range_deg=1, \
                                                                            translate_range=0)
                                        pseudo_cams.append(PseudoCamera(
                                            R=tgt_pose[pseudo_cams[-1].uid][:3, :3], T=tgt_pose[pseudo_cams[-1].uid][:3, 3],
                                            FoVx=pseudo_cams[-1].FoVx, FoVy=pseudo_cams[-1].FoVy,
                                            image_name=pseudo_cams[-1].image_name, uid=pseudo_cams[-1].uid+100*i,
                                            cam_uid=pseudo_cams[-1].uid, image=None, gt_alpha_mask=None, 
                                            width=pseudo_cams[-1].image_width, height=pseudo_cams[-1].image_height, 
                                            data_device=args.data_device
                                        ))    # 存储插值视角
                                    slerp_and_warp_cams.append(pseudo_cams)

                                slerp_cams.append(slerp_and_warp_cams)

                    if len(pseudo_views) == 2:
                        # 随机取两个外参之间的视角，然后这两个视角互相监督
                        for i in range(len(slerp_point_cam)):
                            for j in range(i + 1, len(slerp_point_cam)):
                                pseudo_poses = generate_interpolated_poses([slerp_point_cam[i],
                                                            slerp_point_cam[j]]) # [2, 4, 4]
                                
                                pseudo_cams = list()
                                for k in range(len(pseudo_poses)):
                                    pseudo_view = slerp_point_cam[i]
                                    pseudo_cams.append(PseudoCamera(
                                        R=pseudo_poses[k][:3, :3], T=pseudo_poses[k][:3, 3],
                                        FoVx=pseudo_view.FoVx, FoVy=pseudo_view.FoVy,
                                        image_name=pseudo_view.image_name, uid=pseudo_view.uid+100*i,
                                        cam_uid=pseudo_view.uid, image=None, gt_alpha_mask=None, 
                                        width=pseudo_view.image_width, height=pseudo_view.image_height, 
                                        data_device=args.data_device
                                    ))    # 存储插值视角
                                slerp_cams.append(pseudo_cams)

                    self.slerp_cameras[resolution_scale] = slerp_cams
            #--------------------zzt--------------------
        # print(f'loaded_iter:{self.loaded_iter}') # loaded_iter:None
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            if init_with_gt_depth:
                points, colors, normals, opa = generate_ply_from_depth(scene_info.point_cloud, self.train_cameras[1.0], r=args.resolution)
                pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
                self.gaussians.create_from_pcd_with_opa(pcd, self.cameras_extent, opa)
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        
    def save(self, iteration, color=None):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        if color is not None:
            self.gaussians.save_ply_color(os.path.join(point_cloud_path, "point_cloud_color.ply"), color)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getEvalCameras(self, scale=1.0):
        return self.eval_cameras[scale]
    
    #--------------------zzt--------------------
    def getTgtCameras(self, scale=1.0):
        return self.target_cameras[scale]
    
    def getSlerpCameras(self, scale=1.0):
        return self.slerp_cameras[scale]
    
    def getIntrinsic(self, scale=1.0):  #返回相机的内参矩阵，可以根据指定的缩放因子 scale 获取相应分辨率的相机内参矩阵。
        return self.intrinsic[scale]
    #--------------------zzt--------------------
    



class RenderScene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, spiral=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.test_cameras = {}

        if 'scan' in args.source_path:
            scene_info = sceneLoadTypeCallbacks["SpiralDTU"](args.source_path)
        else:
            scene_info = sceneLoadTypeCallbacks["Spiral"](args.source_path)
        
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Render Cameras", resolution_scales)
            self.test_cameras[resolution_scale] = renderCameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            pass


    def getRenderCameras(self, scale=1.0):
        return self.test_cameras[scale]
