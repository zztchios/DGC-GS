import numpy as np
import os
import cv2
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, \
    read_intrinsics_binary, read_extrinsics_binary, qvec2rotmat
from utils.graphics_utils import focal2fov
from typing import NamedTuple
from PIL import Image
from utils import pose_utils
from scene import RenderScene, Scene, GaussianModel
import torch
from utils.general_utils import PILtoTorch
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, renderCameraList_from_camInfos
from utils.depth_warping import depth_warp_tensor_scale
from utils.loss_utils import l1_loss, ssim
import re

class CameraInfo(NamedTuple):
    uid: str
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth_mono: np.array


def generateLLFFCameras(poses):
    cam_infos = []
    Rs, tvecs, height, width, focal_length_x = pose_utils.convert_poses(poses) 
    # print(Rs, tvecs, height, width, focal_length_x)
    for idx, _ in enumerate(Rs):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(Rs)))
        sys.stdout.flush()

        uid = idx
        R = np.transpose(Rs[idx])
        T = tvecs[idx]

        FovY = focal2fov(focal_length_x, height)
        FovX = focal2fov(focal_length_x, width)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=None, depth_mono=None, 
                              image_path=None, image_name=None, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readColmapCameras(basedir):
    try:
        cameras_extrinsic_file = os.path.join(basedir, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(basedir, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(basedir, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(basedir, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        
        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported but found {}!".format(intr.model)
        reading_dir = "images"
        images_folder = os.path.join(basedir, reading_dir)
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        depth_mono_path = os.path.join('/'.join(images_folder.split("/")[:-1]), 'depth_maps_dpt', 'depth_' + os.path.basename(extr.name).split(".")[0] + '.png')
        depth_mono = Image.open(depth_mono_path)

        cam_info = CameraInfo(uid=image_path, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth_mono=depth_mono,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        
        cam_infos.append(cam_info)
    sys.stdout.write('\n')

    # print("intr", intr.params[0])  
    # ([3.26052633e+03, 2.01600000e+03, 
    # 1.51200000e+03, 1.97504638e-02])
    # print("cam_infos length:", len(cam_infos), cam_infos[0].R, cam_infos[0].T)
    return cam_infos


def get_llff_poses(basedir):
    # Load poses and bounds.
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    print('Loaded poses', poses_arr.shape)  # 20, 17  images=20
    poses_o = poses_arr[:, :-2].reshape([-1, 3, 5]) # pose matrix
    bounds = poses_arr[:, -2:]                      # 视角 到 场景的最近和最远距离
    # print(poses_o[0], bounds[0])
    # Pull out focal length before processing poses.
    # Correct rotation matrix ordering (and drop 5th column of poses).
    fix_rotation = np.array([
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
                            dtype=np.float32)
    inv_rotation = np.linalg.inv(fix_rotation)
    # 将LLFF的相机坐标系变成OpenGL/NeRF的相机坐标系
    poses = poses_o[:, :3, :4] @ fix_rotation   # RT
    H, W, f = poses_o[:1, :3, 4:][0]
    
    print(H, W, f)
    print(bounds[0])
    print(poses_o[:1, :3, 4:].shape)
    # Recenter poses.
    render_poses = pose_utils.recenter_poses(poses)

    # Separate out 360 versus forward facing scenes.
    render_poses = pose_utils.generate_spiral_path(
          render_poses, bounds, n_frames=180)
    render_poses = pose_utils.backcenter_poses(render_poses, poses)
    render_poses = render_poses @ inv_rotation
    render_poses = np.concatenate([render_poses, np.tile(poses_o[:1, :3, 4:], (render_poses.shape[0], 1, 1))], -1)
    render_cam_infos = generateLLFFCameras(render_poses.transpose([1,2,0]))
    print(np.tile(poses_o[:1, :3, 4:], (render_poses.shape[0], 1, 1)).shape)


def setup_stereo_matching(source_path, name, source_image, target_image, K, R, T):
    depth_dir = os.path.join(source_path, 'depth', name+'.jpg')
    disp_dir = os.path.join(source_path, 'disp', name+'.jpg')
    target_g = os.path.join(source_path, 'target_g', name+'.jpg')

    # convert images to grayscale
    source_image = source_image.cpu().numpy()
    target_image = target_image.cpu().numpy()
    

    
    if source_image.dtype != np.uint8:
        source_image_g = (source_image * 255).astype(np.uint8)
    source_image_g = np.transpose(source_image_g, (1, 2, 0))
    source_image_g = cv2.cvtColor(source_image_g.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    if target_image.dtype != np.uint8:
        target_image_g = (target_image * 255).astype(np.uint8)
    target_image_g = np.transpose(target_image_g, (1, 2, 0))
    target_image_g = cv2.cvtColor(target_image_g.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # target_image_g = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)
    # print(target_image_g.shape, source_image.dtype)
    # target_image_g = cv2.cvtColor(target_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # target_image_g = cv2.cvtColor(target_image.cpu().numpy(), cv2.COLOR_BGR2GRAY)
    # 将内参K转换为OpenCV期望的形式
    K = K.cpu().numpy().astype(np.float64)
    R = R.cpu().numpy().astype(np.float64)
    T = T.cpu().numpy().astype(np.float64)
    # 构建基本矩阵(F矩阵)和单应性矩阵(H矩阵)，但在立体匹配中我们更关注基本矩阵
    # 这里简化处理，直接从旋转和平移构造基本矩阵所需的参数
    # E = cv2.findEssentialMat(R=R, T=T, cameraMatrix=K, method=cv2.RANSAC)
    # F, mask = cv2.findFundamentalMat(points1=None, points2=None, essentialMatrix=E, method=cv2.FM_RANSAC)
    # 步骤2: 立体校正
    
    zeros_vec = np.zeros((5,1)).astype(np.float64)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, zeros_vec, K, zeros_vec, (source_image_g.shape[1], source_image_g.shape[0]), R, T)

    # 步骤3: 生成映射
    map_x1, map_y1 = cv2.initUndistortRectifyMap(K, None, R1, P1, (source_image.shape[2], source_image.shape[1]), cv2.CV_32FC1)
    map_x2, map_y2 = cv2.initUndistortRectifyMap(K, None, R2, P2, (source_image.shape[2], source_image.shape[1]), cv2.CV_32FC1)
    
    # 获取相机矩阵P1和P2
    # P1 = K * [I|0], P2 = K * [R|T]
    # print("K", K)
    # print((K @ R).shape)
    # print((K @ T).shape)
    # P1 = np.hstack((K, np.zeros((3, 1))))
    # P2 = np.hstack((K @ R, K @ T))
    
    # 使用映射重投影图像
    rec1 = cv2.remap(source_image_g, map_x1, map_y1, cv2.INTER_LINEAR)
    rec2 = cv2.remap(target_image_g, map_x2, map_y2, cv2.INTER_LINEAR)

    # 初始化StereoBM对象进行块匹配
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    
    # 设置左、右相机的投影矩阵
    # stereo.setLeftRightRectificationMaps(P1, P2)
    # 执行立体匹配
    disparity = stereo.compute(rec1, rec2)
    
    # 步骤5: 计算深度图
    # 基线距离，即 T 的前两个元素表示从左相机到右相机的水平和垂直距离
    # print("T shape:", T.shape)
    baseline = np.sqrt(T[0]**2 + T[1]**2)
    f = K[0, 0]  # 假设 fx = fy
    # 将视差转换为深度
    depth = baseline * f / (disparity+1e-8)
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    disp_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    # print(depth_normalized.max(), depth_normalized.min())
    cv2.imwrite(depth_dir, depth_normalized.astype(np.uint8))
    cv2.imwrite(disp_dir, disp_normalized.astype(np.uint8))
    cv2.imwrite(target_g, target_image_g.astype(np.uint8))
    return disparity


def mask_depth(basedir):
    try:
        cameras_extrinsic_file = os.path.join(basedir, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(basedir, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(basedir, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(basedir, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    benchmark = "LLFF" # or "DTU"

    if benchmark=="DTU":
        scenes = ["scan30", "scan34", "scan41", "scan45",  "scan82", "scan103", "scan38", "scan21", "scan55", "scan40", "scan63", "scan31", "scan8", "scan110", "scan114"]
    elif benchmark=="LLFF":
        scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]

    N_sparse = 3
    train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
    exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
    test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]
    if N_sparse > 0:
        train_idx = train_idx[:N_sparse]

    reading_dir = "images"
    cam_infos_unsorted = readColmapCameras(basedir)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_idx]
    test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_idx]
    eval_cam_infos = test_cam_infos
    print('train', [info.image_path for info in train_cam_infos])
    print('eval', [info.image_path for info in eval_cam_infos])
    target_width = 398
    target_height = 298
    i = 0
    for info in train_cam_infos:
        file_save_path = os.path.join(basedir, "images_resize", info.image_name+".png")
        mask_save_path = os.path.join(basedir, "mask_resize", "mask_"+info.image_name+".png")

        match = re.search(r'_(\d+)', info.image_name)  # 修改正则表达式，只匹配第一个下划线后面的数字
        if match:
            number_str = match.group(1)
            number_int = int(number_str)
            
        mask_path = os.path.join(basedir, "mask", f"{(number_int-1):03d}.png")
        print(mask_path)
        i=i+1
        # 打开图像
        with Image.open(info.image_path) as img:
            # 调整图像大小
            resized_img = img.resize((target_width, target_height), Image.ANTIALIAS)
            # # bg_mask = (resized_img.max(0, keepdim=True).values < 15/255)
            # resized_image_torch = torch.from_numpy(np.array(resized_img)) / 255.0
            # resized_image_torch = resized_image_torch.permute(2, 0, 1)
            # bg_mask = (resized_image_torch.max(0, keepdim=True).values < 15/255)
            # bg_mask_clone = bg_mask.clone()
            # for i in range(1, 50):
            #     bg_mask[:, i:] *= bg_mask_clone[:, :-i] 
            # # print(resized_img.size)
            # # 保存调整大小后的图像到同一文件夹

            # bg_mask = np.where(bg_mask[0] > 0, 0, 255).astype(np.uint8)
            # print(bg_mask)
            # bg_mask = Image.fromarray(bg_mask)
            resized_img.save(file_save_path)
        # print(file_save_path, mask_save_path)
        with Image.open(mask_path) as mask:
            # 调整图像大小
            resized_mask = mask.resize((target_width, target_height), Image.ANTIALIAS)
            resized_mask.save(mask_save_path)

def resize_file(dirfile):
    dir_file =  os.path.join(dirfile, 'images')
    target_width = 398
    target_height = 298
    for filename in os.listdir(dir_file):
        file_save_path = os.path.join(dirfile, "images_resize", os.path.basename(filename))
        print(file_save_path)
        with Image.open(os.path.join(dir_file, filename)) as img:
            # 调整图像大小
            resized_img = img.resize((target_width, target_height), Image.ANTIALIAS)
            resized_img.save(file_save_path)

def get_RT_scale(source_path):
    scene_info = sceneLoadTypeCallbacks["Spiral"](source_path)
    test_cameras = renderCameraList_from_camInfos(scene_info.test_cameras, 4, args)
    # for cam in test_cameras:
    #     print(cam.uid, cam.R, cam.T)
    return test_cameras
    
    


if __name__ == '__main__':
    # intr, cam_infos = readColmapCameras('/media/pc/D/zzt/depth_3DGS/datasets/llff/fern')
    # get_llff_poses('/media/pc/D/zzt/depth_3DGS/datasets/llff/fern')
    
    from arguments import ModelParams
    from argparse import ArgumentParser, Namespace
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.source_path = '/media/pc/D/zzt/depth_3DGS/datasets/dtu/scan40'
    # args.source_path = '/media/pc/D/zzt/depth_3DGS/datasets/llff/fern'
    args.eval = True
    args.resolution = 8
    import random
    '''
    dataset = lp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree) #（重点看，需要转跳）创建一个 GaussianModel 类的实例，输入一系列参数，其参数取自数据集。
    scene = Scene(dataset, gaussians)
    scene_sprical = RenderScene(dataset, gaussians, spiral=True)
    
    viewpoint_stack = scene.getTrainCameras().copy()
    print(len(viewpoint_stack))
    test_cameras = get_RT_scale(args.source_path).copy()
    print(len(test_cameras))
    len_view = len(viewpoint_stack)
    viewpoint_cam_tgt = viewpoint_stack.pop()
    viewpoint_stack_s = viewpoint_stack
    scale = {}
    Loss_1 = {}
    for s in np.linspace(0, 30, 21):
        viewpoint_stack = viewpoint_stack_s.copy()
        for i in range(len(viewpoint_stack)):
            viewpoint_cam_source = viewpoint_stack.pop()
            K = scene.getIntrinsic()[viewpoint_cam_source.uid]
            R_A = test_cameras[i+1].R
            T_A = test_cameras[i+1].T
            R_B = test_cameras[0].R

            T_B = test_cameras[0].T

            # Calculate the transformation from A to B
            # R_A = torch.tensor(R_A.T, dtype=torch.float32).cuda()
            # R_B = torch.tensor(R_B.T, dtype=torch.float32).cuda()
            # T_A = torch.tensor(T_A, dtype=torch.float32).cuda()
            # T_B = torch.tensor(T_B, dtype=torch.float32).cuda()
            # R_AB = R_B @ torch.inverse(R_A)
            # T_AB = T_B - (R_AB @ T_A)
            # resolution = 4
            # image_source = PILtoTorch(viewpoint_cam_source.original_image, resolution)
            # image_target = PILtoTorch(viewpoint_cam_tgt.original_image, resolution)
            image_source = viewpoint_cam_source.original_image
            image_target = viewpoint_cam_tgt.original_image
            depth_mono = viewpoint_cam_source.depth
            
            warp_image = depth_warp_tensor_scale(img_tensor=viewpoint_cam_source.original_image, depth_tensor=depth_mono, K=K, R_A=R_A, T_A=T_A, \
                                                R_B=R_B, T_B=T_B, s=s)
            # print(warp_image.shape, image_target.shape)
            loss = l1_loss(warp_image, image_target)
            # print(viewpoint_cam_source.image_name)
            loss += (1.0 - 0.2) * loss + 0.2 * (1.0 - ssim(warp_image, image_target))
            if len(Loss_1) < len_view:
                Loss_1[viewpoint_cam_source.uid]=loss
                scale[viewpoint_cam_source.uid]=s
            else:
                if loss < Loss_1[viewpoint_cam_source.uid]:
                    # print(loss, Loss_1[i])
                    # print(viewpoint_cam_source.uid, s)
                    Loss_1[viewpoint_cam_source.uid] = loss
                    scale[viewpoint_cam_source.uid] = s
    '''
    # for key, value in Loss_1.items():
    #     print(f"uid: {key}, Loss: {value}")

    # for key, value in scale.items():
    #     print(f"uid: {key}, Value: {value}")
                     
                
    
    mask_depth(args.source_path,)
    # resize_file('/media/pc/D/zzt/depth_3DGS/datasets/dtu/scan40',)




    
        # disparity = setup_stereo_matching(args.source_path, viewpoint_cam_source.image_name, \
        #                                   image_source, image_target, \
        #                                   K, R_AB, T_AB)
        
    # # 示例内参矩阵、旋转矩阵、平移向量（这些值需要根据实际情况填写）
    # K = np.array([[f1, 0, cx], [0, f2, cy], [0, 0, 1]])  # 这里f1, f2, cx, cy需要替换为实际焦距和光心坐标
    # R = np.eye(3)  # 示例旋转矩阵，实际情况下应使用非单位矩阵
    # T = np.array([tx, ty, tz])  # 示例平移向量，tx, ty, tz需替换为实际数值

    # # 假设你已经有两幅图像
    # source_image = cv2.imread('path_to_source_image.jpg', cv2.IMREAD_GRAYSCALE)
    # target_image = cv2.imread('path_to_target_image.jpg', cv2.IMREAD_GRAYSCALE)

    # # 执行立体匹配
    # disparity = setup_stereo_matching(source_image, target_image, K, R, T)

    # # 显示视差图
    # cv2.imshow('Disparity Map', disparity / 16)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
