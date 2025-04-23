from dataclasses import dataclass
from pathlib import Path
from typing_extensions import Dict, Literal, Optional

import os
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch
import tyro
from PIL import Image
from utils.depth_from_pretrain import depth_from_pretrain
# from utils.depth_from_pretrain_dnsplatting import depth_from_pretrain
from utils.utils import depth_path_to_tensor, get_filename_list, save_depth
from utils.pose_utils import (read_cameras_binary,
                              qvec2rotmat, 
                              read_cameras_text,
                              read_images_binary,
                              read_images_text,
                              read_points3d_binary,
                              read_points3D_text
                              )
from rich.console import Console
from rich.progress import track
from torch import Tensor
import matplotlib
# from colmap_depth import read_array
from skimage.transform import resize

CONSOLE = Console(width=120)
BATCH_SIZE = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    # print("array:", width, height)
    return np.transpose(array, (1, 0, 2)).squeeze()

def get_geometric_bin_files(directory):
    """
    获取指定目录下所有以 '.geometric.bin' 结尾的文件路径

    :param directory: 目标目录
    :return: 文件路径列表
    """
    geometric_bin_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.geometric.bin'):
                geometric_bin_files.append(os.path.join(root, file))
    return geometric_bin_files

def bin2depth(recon_dir, output_dir):
    # depth_map = '0.png.geometric.bin'
    # print(depthdir)
    # if min_depth_percentile > max_depth_percentile:
    #     raise ValueError("min_depth_percentile should be less than or equal "
    #                      "to the max_depth_perceintile.")

    # Read depth and normal maps corresponding to the same image.
    
    for im_data in track(get_geometric_bin_files(recon_dir), description="processing..."): 
        if not os.path.exists(im_data):
            raise FileNotFoundError("file not found: {}".format(im_data))
        depth_map = read_array(im_data)
        os.makedirs(output_dir, exist_ok=True)                

        min_depth, max_depth = np.percentile(
            depth_map[depth_map>0], [5, 95])

        depth_map[depth_map <= 0] = np.nan # 把0和负数都设置为nan，防止被min_depth取代
        depth_map[depth_map < min_depth] = min_depth
        depth_map[depth_map > max_depth] = max_depth

        # 分离文件名和扩展名
        # name_part = os.path.basename(im_data)
        name_part = Path(im_data).stem

        # 检查文件名是否包含 .JPG 或 .jpg
        if '.JPG' in name_part or '.jpg' in name_part:
            # 提取 .JPG 或 .jpg 之前的部分
            base_name = name_part.split('.')[0]
            
            # mask_filename = f"{base_name}.png"
            # depth_mask = gen_mask(np.nan_to_num(depth_map))
            # depth_mask = resize(depth_mask, (3024//8, 4032//8))
            # os.makedirs(os.path.join(os.path.dirname(os.path.dirname(depthdir)), 'mask'), exist_ok=True)
            # plt.imsave(os.path.join(os.path.dirname(os.path.dirname(depthdir)), 'mask', mask_filename), depth_mask, cmap='gray')

            new_filename = f"{base_name}.npy"
            new_imgname = f"{base_name}.png"
            # 构建新的文件路径
            save_npy_path = os.path.join(output_dir, new_filename)

        # np.save(save_npy_path, depth_map)
        depth_npy = depth_map
        depth_npy = np.nan_to_num(depth_npy) # nan全都变为0

        # 执行缩放操作
        depth_npy = resize(depth_npy, (3024//8, 4032//8))

        depth_map = (depth_npy - depth_npy.min()) * 255.0 / (depth_npy.max() - depth_npy.min()) 

        depth_map = depth_map.astype(np.uint8)
        
        image = Image.fromarray(depth_map).convert('L')

        image = image.resize((4032//8, 3024//8), resample=Image.Resampling.LANCZOS)
        # print(save_npy_path, depthdir + f"rect_{i:03d}_3_r5000" + '.png')
        # print(os.path.join(os.path.dirname(os.path.dirname(depthdir)), 'depth_pho_colmap_npy'))
        np.save(save_npy_path, depth_npy)

        # 构建新的文件路径
        save_dep_path = os.path.join(output_dir, new_imgname)
        # print(f'bin2depth path:{save_dep_path}, {save_npy_path}')
        # image_array = np.array(image)

        # # 打印特定位置的像素值
        # print(image_array.shape)

        image.save(save_dep_path)


def gen_mask(depth_gt, depth):

    # 步骤1：过滤掉所有小于1的值，并保持它们的索引
    # print(f'depth_gt:{depth_gt.shape}, {depth.shape}')

    valid_indices = (depth > 0) & (depth_gt > 15) # & (depth_gt < 40)

    valid_values = depth[valid_indices]

    percentiles = torch.tensor([0, 50], dtype=torch.float32) / 100.0  # 将百分比转换为小数
    threshold_value_min, threshold_value_max = torch.quantile(valid_values, percentiles)
    # 步骤3：计算有效元素的数量和小于阈值的有效元素数量
    total_valid_elements = len(valid_values)
    # print(f'total_valid:{total_valid_elements}, {np.sum(threshold_value_min < valid_values)}, {threshold_value_min}, {np.min(depth)}')
    below_threshold_indices =  (threshold_value_min < valid_values) & (valid_values < threshold_value_max)
    num_below_threshold = torch.sum(below_threshold_indices)
    
    target_percentage = 20

    target_num_below_threshold = max(1, int(total_valid_elements * target_percentage / 100))

    if num_below_threshold >= target_num_below_threshold:
        # 如果满足条件，则创建mask
        mask = torch.zeros_like(depth, dtype=bool)
        # print(f'below_threshold_indices:{below_threshold_indices.shape}')
        valid_positions = torch.nonzero(valid_indices, as_tuple=True)
        mask_indices = (valid_positions[0][below_threshold_indices], valid_positions[1][below_threshold_indices])
        mask[mask_indices] = True
    else:
        # 步骤2：对剩下的值进行排序（从小到大）
        sorted_indices = torch.argsort(valid_values)

        # 计算需要选择多少个最小的元素作为有用的mask
        num_elements_to_select = max(1, int(total_valid_elements * target_percentage / 100))
        print(f'less_mask: {num_elements_to_select}, {depth.min()}, {depth.max()}')
        # 步骤3和4：创建一个全为False的布尔数组，长度与原始数组相同
        mask = torch.zeros_like(depth, dtype=bool)

        # 获取原始二维数组中有效值的位置
        valid_positions = torch.nonzero(valid_indices, as_tuple=True)
        print(f'num_elements_to_select:{num_elements_to_select}')
        # 将最小的30%的值的位置设置为True，在原始数组中的对应位置也设置为True
        smallest_percent_indices = sorted_indices[num_elements_to_select*0.1:num_elements_to_select*1.1]
        mask_indices = (valid_positions[0][smallest_percent_indices],
                        valid_positions[1][smallest_percent_indices])
        mask[mask_indices] = True
    
    return mask

def remove_3_views_from_path(path):
    parts = path.split(os.sep)
    if '3_views' in parts:
        parts.remove('3_views')
    return os.sep.join(parts)

@dataclass
class ColmapToAlignedMonoDepths:
    """Converts COLMAP dataset SfM points to scale aligned mono-depth estimates

    COLMAP dataset is expected to be in the following form:
    <data>
    |---image_path
    |   |---<image 0>
    |   |---<image 1>
    |   |---...
    |---colmap
        |---sparse
            |---0
                |---cameras.bin
                |---images.bin
                |---points3D.bin

    This function provides the following directories in that <data> root
    |---sfm_depths
    |   |---<sfm_depth 0>
    |   |---<sfm_depth 1>
    |   |---...
    |---mono_depth
    |   |---<mono_depth 0>.png
    |   |---<mono_depth 0>_aligned.npy
    """

    data: Path
    """Input dataset path"""
    sparse_path: Path = Path("3_views/dense/sparse")
    """Input dense dataset path"""
    dense_3views_path: Path = Path("3_views/dense/stereo/depth_maps")
    dense_path: Path = Path("dense/stereo/depth_maps")
    """Default path of colmap sparse dir"""
    img_dir_name: str = "3_views/images"
    """Directory name of where input images are stored. Default is '/images', but you can remap it to something else. """
    mono_depth_network: Literal["zoe"] = "zoe"
    """What sparse 3_view colmap to use"""
    skip_sparse_colmap_to_depths: bool = False
    """What mono depth network to use"""
    skip_colmap_to_depths: bool = False
    """What mono depth network to use"""
    skip_3view_dense_depths: bool = False
    """What mono depth network to use"""
    skip_dense_colmap_to_depths: bool = False
    """Skip colmap to sfm step"""
    skip_mono_depth_creation: bool = False
    """Skip mono depth creation"""
    skip_alignment: bool = False
    """Skip alignment"""
    skip_patch: bool = False
    """Skip patch"""
    patch: int = 63 # 126 
    """Number of patch to align depths"""
    iterations: int = 1000
    """Number of grad descent iterations to align depths"""
    align_method: Literal["closed_form", "grad_descent"] = "closed_form"
    """Use closed form solution for depth alignment or graident descent"""

    def main(self) -> None:
        sfm_depth_path = self.data / Path("sfm_depths")
        sfm_dense_path = self.data / Path("dense_depths")
        
        # 这个不行，不建议使用
        if self.skip_sparse_colmap_to_depths:
            CONSOLE.print("Generating sfm depth maps from sparse colmap reconstruction")
            colmap_sfm_points_to_depths(
                recon_dir=self.data / self.sparse_path,
                output_dir=sfm_depth_path,
                include_depth_debug=True,
                input_images_dir=self.data / self.img_dir_name,
            )
            sfm_path = sfm_depth_path

        # 预训练的深度估计
        if not self.skip_mono_depth_creation:
            CONSOLE.print("Computing mono depth estimates")
            if not (self.data / Path("mono_depth")).exists() or True:
                depth_from_pretrain(
                    input_folder=self.data,
                    img_dir_name=self.img_dir_name,
                    path_to_transforms=None,
                    create_new_transforms=False,
                    is_euclidean_depth=False,
                    pretrain_model=self.mono_depth_network,
                )
            else:
                CONSOLE.print("Found previous /mono_depth path")
        
        # 稠密的colmap深度图
        if not self.skip_dense_colmap_to_depths:
            CONSOLE.print("Generating sfm depth maps from dense colmap reconstruction")
            if not self.skip_3view_dense_depths:
                recon_dir = self.data / self.dense_path
            else:
                recon_dir = self.data / self.dense_3views_path

            bin2depth(recon_dir=recon_dir,
                      output_dir=sfm_dense_path)
            sfm_path = sfm_dense_path

        if not self.skip_alignment:
            CONSOLE.print("Aligning sparse depth maps with mono estimates")
            # Align sparse sfm depth maps with mono depth maps
            batch_size = BATCH_SIZE

            sfm_depth_filenames = get_filename_list(
                image_dir=sfm_path, ends_with=".npy"
                # image_dir=self.data / Path("sfm_depths"), ends_with=".npy"
            )

            mono_depth_filenames = get_filename_list(
                image_dir=self.data / Path("mono_depth"), ends_with=".npy"
                # image_dir=self.data / Path("mono_anything_depth"), ends_with=".npy"
            )
            # filter out aligned depth and frames not have pose
            sfm_name = [item.name for item in sfm_depth_filenames]

            mono_depth_filenames = [
                item
                for item in mono_depth_filenames
                if "_aligned.npy" not in item.name and str(item.stem) in str(sfm_name)
            ]

            assert len(sfm_depth_filenames) == len(mono_depth_filenames)

            H, W = depth_path_to_tensor(sfm_depth_filenames[0]).shape[:2]

            num_frames = len(sfm_depth_filenames)
            for batch_index in range(0, num_frames, batch_size):
                batch_sfm_frames = sfm_depth_filenames[
                    batch_index : batch_index + batch_size
                ]

                batch_mono_frames = mono_depth_filenames[
                    batch_index : batch_index + batch_size
                ]

                with torch.no_grad():
                    mono_depth_tensors = []
                    sparse_depths = []

                    for frame_index in range(len(batch_sfm_frames)):
                        sfm_frame = batch_sfm_frames[frame_index]
                        mono_frame = batch_mono_frames[frame_index]
                        # print(f'mono_frame:{mono_frame}, sfm_frame:{sfm_frame}')
                        mono_depth = depth_path_to_tensor(
                            mono_frame,
                            return_color=False,
                            scale_factor=0.001 if mono_frame.suffix == ".png" else 1,
                        )  # note that npy depth maps are in meters
                        # print(f'mono_depth:{mono_depth.shape}')
                        if not self.skip_mono_depth_creation:
                            mono_depth = F.interpolate(mono_depth.squeeze(2).unsqueeze(0).unsqueeze(0), 
                                                       size= (mono_depth.shape[0] // 8, 
                                                              mono_depth.shape[1] // 8), 
                                                       mode='bilinear', 
                                                       align_corners=False).squeeze(0).squeeze(0).unsqueeze(2)
                        
                        mono_depth_tensors.append(mono_depth)

                        sfm_depth = depth_path_to_tensor(
                            sfm_frame, return_color=False, 
                            scale_factor=0.001 if mono_frame.suffix == ".png" else 1,
                        )
                        
                        if not self.skip_colmap_to_depths:
                            sfm_depth = F.interpolate(sfm_depth.squeeze(2).unsqueeze(0).unsqueeze(0), 
                                                    size= (sfm_depth.shape[0] // 8, 
                                                           sfm_depth.shape[1] // 8), 
                                                    mode='bilinear', 
                                                    align_corners=False).squeeze(0).squeeze(0).unsqueeze(2)
                        # print(mono_depth.shape, sfm_depth.shape)
                        sparse_depths.append(sfm_depth)

                    mono_depth_tensors = torch.stack(mono_depth_tensors, dim=0)
                    sparse_depths = torch.stack(sparse_depths, dim=0)
                    
                if self.align_method == "closed_form":
                    # mask = (sparse_depths > 0.1) & (sparse_depths < 10.0)
                    # for batch_sfm_frames 
                    mask = []
                    for frame_index in range(len(batch_sfm_frames)):
                        # sfm_frame = batch_sfm_frames[frame_index]
                        # mono_frame = batch_mono_frames[frame_index]
                        # print(sparse_depths[frame_index].shape, mono_depth_tensors[frame_index].shape)
                        mask.append(gen_mask(sparse_depths[frame_index], mono_depth_tensors[frame_index]))

                        base_name = Path(sfm_frame).stem
                        base_name = base_name.split('.')[0]
                        mask_filename = f"{base_name}.png"
                        # mask_path = Path(sfm_dense_path) / Path('mask') / Path(mask_filename)
                        mask_path = Path(self.data) / Path('mask') / Path(mask_filename)
                        # print(f'mask_path:{mask_path.parent}')
                        mask_path.parent.mkdir(parents=True, exist_ok=True)

                        plt.imsave(mask_path, mask[-1].squeeze(2).cpu().numpy(), cmap='gray')
                    
                    mask = torch.stack(mask)
                    # print(f'mask:{mask.shape}')
                    if not self.skip_patch:

                        depth_aligned = align_neighbor_scale(sparse_depths,
                                             mono_depth_tensors,
                                             mask,
                                             self.patch)

                        # print(f'sparse:{sparse_depths[0][27][471]}, \
                        #       mono:{mono_depth_tensors[0][27][471]}, \
                        #       align:{depth_aligned[0][27][471]}, \
                        #       {sparse_depths[0][226][358]}, \
                        #       {mono_depth_tensors[0][226][358]}, \
                        #       {depth_aligned[0][226][358]}')
                        
                        mse_loss = torch.nn.MSELoss()
                        avg = mse_loss(depth_aligned[mask], sparse_depths[mask])    
                        CONSOLE.print(
                            f"[bold yellow]Average depth alignment error for batch depths is: {avg:3f} which is {'good' if avg<0.2 else 'bad'}"
                        )
                    else:
                        scale, shift = compute_scale_and_shift(
                            mono_depth_tensors, sparse_depths, masks=mask
                        )
                        scale = scale.unsqueeze(1).unsqueeze(2)
                        shift = shift.unsqueeze(1).unsqueeze(2)
                        depth_aligned = scale * mono_depth_tensors + shift

                        # print(f'sparse:{sparse_depths[0][27][471]}, \
                        #       mono:{mono_depth_tensors[0][27][471]}, \
                        #       align:{depth_aligned[0][27][471]}, \
                        #       {sparse_depths[0][226][358]}, \
                        #       {mono_depth_tensors[0][226][358]}, \
                        #       {depth_aligned[0][226][358]}')
                        
                        mse_loss = torch.nn.MSELoss()

                        avg = mse_loss(depth_aligned[mask], sparse_depths[mask])

                        CONSOLE.print(
                            f"[bold yellow]Average depth alignment error for batch depths is: {avg:3f} which is {'good' if avg<0.2 else 'bad'}"
                        )

                elif self.align_method == "grad_descent":
                    depth_aligned = grad_descent(
                        mono_depth_tensors, sparse_depths, iterations=self.iterations
                    )
                
                '''
                row, col:4, 6, 0, tensor([[0., 0., 0.]])
                nei_scale35, nei_shift29:tensor([[0.0082, 0.0256, 0.0000]]), tensor([[  8.1400, -34.1553,   0.0000]])
                nei_scale36, nei_shift30:tensor([[0.0262, 0.0416, 0.0019]]), tensor([[-31.9467, -77.5953,  19.2000]])
                nei_scale37, nei_shift31:tensor([[0.0265, 0.0000, 0.0000]]), tensor([[-34.9305,   0.0000,   0.0000]])
                nei_scale55, nei_shift45:tensor([[-0.0161,  0.0000,  0.0000]]), tensor([[63.4641,  0.0000,  0.0000]])
                avg_scale, avg_shift:tensor([[0.0167, 0.0172, 0.0243]]), tensor([[-10.9264, -15.6984, -31.1848]])
                
                '''
                # print(torch.mean(depth_aligned[:,(4*self.patch):((4+1)*self.patch),(7*self.patch):((7+1)*self.patch)]).item(),
                #       torch.mean(mono_depth_tensors[:,(4*self.patch):((4+1)*self.patch),(7*self.patch):((7+1)*self.patch)]).item())

                # print(torch.mean(depth_aligned[:,(4*self.patch):((4+1)*self.patch),(6*self.patch):((6+1)*self.patch)]).item(),
                #       torch.mean(mono_depth_tensors[:,(4*self.patch):((4+1)*self.patch),(6*self.patch):((6+1)*self.patch)]).item())
                
                # save depths
                for idx in track(
                    range(depth_aligned.shape[0]),
                    description="saving aligned depth images...",
                ):
                    depth_aligned_numpy = depth_aligned[idx, ...].squeeze(-1).detach().cpu().numpy()
                    file_name = str(Path(batch_mono_frames[idx]).with_suffix(""))
                    
                    original_path = Path(file_name)
                    new_path = original_path.parent.parent / "npy_depth" / original_path.name
                    new_path2 = original_path.parent.parent / "depth_npy" / original_path.name
                    os.makedirs(original_path.parent.parent / "npy_depth", exist_ok=True) 
                    os.makedirs(original_path.parent.parent / "depth_npy", exist_ok=True)
                    # save only npy
                    np.save(new_path.with_suffix(".npy"), depth_aligned_numpy)
                    np.save(new_path2.with_suffix(".npy"), depth_aligned_numpy)



def align_neighbor_scale(depth_colmap, 
                         depth_est, 
                         mask, 
                         patch_size, 
                         missing_ratio_threshold=0.2):
    b, h, w, _ = depth_colmap.shape
    depth_aligned = torch.zeros_like(depth_colmap)
    for k in range(b):
        
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                block_size = patch_size
                ratio_patch = False
                while True:
                    patch_colmap = depth_colmap[k, max(0, i-block_size):min(h, i+block_size), 
                                                max(0, j-block_size):min(w, j+block_size), ...].unsqueeze(0)
                    patch_est = depth_est[k, max(0, i-block_size):min(h, i+block_size), 
                                                max(0, j-block_size):min(w, j+block_size), ...].unsqueeze(0)
                    patch_mask = mask[k, max(0, i-block_size):min(h, i+block_size), 
                                                max(0, j-block_size):min(w, j+block_size), ...].unsqueeze(0)
                    missing_ratio = torch.count_nonzero(patch_mask) / (patch_size**2)
                    print(torch.count_nonzero(patch_mask))
                    if missing_ratio > missing_ratio_threshold:
                        break
                    elif missing_ratio < 0.1:
                        print(f'i:{i}, j:{j}')  # (346,233)(4,2)
                        block_size += 4*patch_size
                        # ratio_patch = True
                        # break
                    else:
                        block_size += patch_size
                    # print(f'patch_colmap:{patch_colmap.shape}, patch_est:{patch_est.shape}')

                scale, shift = compute_scale_and_shift(patch_est, patch_colmap, masks=patch_mask)
                # print('scale:', scale.shape, shift.shape)
                scale = scale.unsqueeze(1).unsqueeze(2)
                shift = shift.unsqueeze(1).unsqueeze(2)
                
                mono_depth_patch = depth_est[k, i:i+patch_size, j:j+patch_size, ...].unsqueeze(0)

                depth_aligned_patch = scale * mono_depth_patch + shift
                
                # if ratio_patch:
                #     scale_all, shift_all = compute_scale_and_shift(depth_est[k, ..., ..., ...].unsqueeze(0), 
                #                                            depth_colmap[k, ..., ..., ...].unsqueeze(0), 
                #                                            masks=mask[k, ..., ..., ...].unsqueeze(0))
                #     # print("scale_all_patch", scale_all.shape, scale.shape, depth_est[k, ..., ..., ...].shape)
                #     depth_aligned_patch = scale_all * mono_depth_patch + shift_all
                # else:
                #     depth_aligned_all = depth_aligned_patch
                
                # print("depth_aligned_patch", depth_aligned_patch.shape, depth_aligned_all.shape)
                depth_aligned[k, i:i+patch_size, j:j+patch_size, ...] = depth_aligned_patch
                
                # mse_loss = torch.nn.MSELoss()
                # if torch.all(scale.squeeze(1).squeeze(1).ne(0)).item() and torch.sum(mask_patch).item()>100:
                # mask_patch_pop = mask[:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size,...]
                # avg += mse_loss(depth_aligned_patch[mask_patch_pop], patch_colmap[mask_patch_pop])

    return depth_aligned

def neighbor_scale(p_x, p_y, 
                   mono_depth_tensors,
                   mono_depth_patch,
                   patch_size, 
                   sparse_depths_patch, 
                   mask_patch):
    new_depth = []
    new_sparse = []
    new_mask = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            
            if p_x+i >= 0 and p_x+i < mono_depth_tensors.shape[1]//patch_size and p_y+j >= 0 and p_y+j < mono_depth_tensors.shape[2]//patch_size:
                depth_idx = (p_x+i)*mono_depth_tensors.shape[2]//patch_size+(p_y+j)
                # print(p_x, p_y, p_x+i, p_y+j, depth_idx)
                nonzero_counts = torch.count_nonzero(mask_patch[depth_idx], dim=(1, 2, 3)).tolist()
                if any(count < 400 for count in nonzero_counts):
                    continue
                else:
                    new_depth.append(mono_depth_patch[depth_idx])
                    new_sparse.append(sparse_depths_patch[depth_idx])
                    new_mask.append(mask_patch[depth_idx])
            
    if new_depth:
        scale, shift = compute_scale_and_shift(
            new_depth, new_sparse, masks=new_mask)
        return scale, shift
    else:
        for i in range(len(mask_patch[-1])):
            if nonzero_counts[i] < 400:
                scale[i], shift[i] = torch.zeros(1), torch.zeros(1)   
                        
        return torch.zeros(1), torch.zeros(1)
             
    

def compute_scale_and_shift(predictions, targets, masks):
    """
    计算线性分布的权重(scale)和平移(shift)，支持单个图像和多个图像块。
    
    :param predictions: 预测值，形状为 (B, H, W) 或者是一个包含多个 (H, W) 张量的列表
    :param targets: 目标值，形状为 (B, H, W) 或者是一个包含多个 (H, W) 张量的列表
    :param masks: 掩码，形状为 (B, H, W) 或者是一个包含多个 (H, W) 张量的列表
    :return: scale 和 shift 的张量，形状为 (B,)
    """
    print(f'predictions:{predictions.shape}, targets:{targets.shape}, masks:{masks.shape}')
    if isinstance(predictions, list):
        # 如果输入是列表，则将列表中的每个元素视为一个batch
        predictions = torch.cat(predictions, dim=2)
        targets = torch.cat(targets, dim=2)
        masks = torch.cat(masks, dim=2)
        return compute_scale_and_shift_single(predictions, targets, masks)
    else:
        return compute_scale_and_shift_single(predictions, targets, masks)
    

# copy from monosdf
def compute_scale_and_shift_single(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def grad_descent(
    mono_depth_tensors: torch.Tensor,
    sparse_depths: torch.Tensor,
    iterations: int = 1000,
    lr: float = 0.1,
    threshold: float = 0.0,
) -> Tensor:
    """Align mono depth estimates with sparse depths.

    Args:
        mono_depth_tensors: mono depths
        sparse_depths: sparse sfm points
        H: height
        W: width
        iterations: number of gradient descent iterations
        lr: learning rate
        threshold: masking threshold of invalid depths. Default 0.

    Returns:
        aligned_depths: tensor of scale aligned mono depths
    """
    aligned_mono_depths = []
    for idx in track(
        range(mono_depth_tensors.shape[0]),
        description="Alignment with grad descent ...",
    ):
        scale = torch.nn.Parameter(
            torch.tensor([1.0], device=device, dtype=torch.float)
        )
        shift = torch.nn.Parameter(
            torch.tensor([0.0], device=device, dtype=torch.float)
        )

        estimated_mono_depth = mono_depth_tensors[idx, ...].float().to(device)
        sparse_depth = sparse_depths[idx].float().to(device)

        mask = sparse_depth > threshold
        estimated_mono_depth_map_masked = estimated_mono_depth[mask]
        sparse_depth_masked = sparse_depth[mask]

        mse_loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam([scale, shift], lr=lr)

        avg_err = []
        for step in range(iterations):
            optimizer.zero_grad()
            loss = mse_loss(
                scale * estimated_mono_depth_map_masked + shift, sparse_depth_masked
            )
            loss.backward()
            optimizer.step()
            
        avg_err.append(loss.item())
        aligned_mono_depths.append(scale * estimated_mono_depth + shift)

    avg = sum(avg_err) / len(avg_err)
    CONSOLE.print(
        f"[bold yellow]Average depth alignment error for batch depths is: {avg:3f} which is {'good' if avg<0.2 else 'bad'}"
    )
    return torch.stack(aligned_mono_depths, dim=0)


def colmap_sfm_points_to_depths(
    recon_dir: Path,
    output_dir: Path,
    min_depth: float = 0.001,
    max_depth: float = 1000,
    max_repoj_err: float = 2.5,
    min_n_visible: int = 5,
    include_depth_debug: bool = True,
    input_images_dir: Optional[Path] = Path(),
) -> Dict[int, Path]:
    """Converts COLMAP's points3d.bin to sparse depth maps

    Args:
        recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
        output_dir: Path to the output directory.
        verbose: If True, logs progress of depth image creation.
        min_depth: Discard points closer than this to the camera.
        max_depth: Discard points farther than this from the camera.
        max_repoj_err: Discard points with reprojection error greater than this
          amount (in pixels).
        min_n_visible: Discard 3D points that have been triangulated with fewer
          than this many frames.
        include_depth_debug: Also include debug images showing depth overlaid
          upon RGB.

    Returns:
        Depth file paths indexed by COLMAP image id
    """
    depth_scale_to_integer_factor = 1
    # recon_dir_3view = recon_dir
    # recon_dir = remove_3_views_from_path(recon_dir)
    if (recon_dir / "points3D.bin").exists():
        ptid_to_info = read_points3d_binary(recon_dir / "points3D.bin")
        cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
        im_id_to_image = read_images_binary(recon_dir / "images.bin")
    elif (recon_dir / "points3D.txt").exists():
        ptid_to_info = read_points3D_text(recon_dir / "points3D.txt")
        cam_id_to_camera = read_cameras_text(recon_dir / "cameras.txt")
        im_id_to_image = read_images_text(recon_dir / "images.txt")
    # Only support first camera
    CAMERA_ID = 1
    W = cam_id_to_camera[CAMERA_ID].width
    H = cam_id_to_camera[CAMERA_ID].height

    iter_images = iter(im_id_to_image.items())
    image_id_to_depth_path = {}

    for im_id, im_data in track(iter_images, description="..."):
        # if 
        # TODO(1480) BEGIN delete when abandoning colmap_parsing_utils
        pids = [pid for pid in im_data.point3D_ids if pid != -1]
        xyz_world = np.array([ptid_to_info[pid].xyz for pid in pids])
        # delete
        # xyz_world = np.array([p.xyz for p in ptid_to_info.values()])
        rotation = qvec2rotmat(im_data.qvec)
        
        z = (rotation @ xyz_world.T)[-1] + im_data.tvec[-1]
        
        errors = np.array([ptid_to_info[pid].error for pid in pids])
        n_visible = np.array([len(ptid_to_info[pid].image_ids) for pid in pids])
        uv = np.array(
            [
                im_data.xys[i]
                for i in range(len(im_data.xys))
                if im_data.point3D_ids[i] != -1
            ]
        )

        idx = np.where(
            (z >= min_depth)
            & (z <= max_depth)
            & (errors <= max_repoj_err)
            & (n_visible >= min_n_visible)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < W)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < H)
        )
        z = z[idx]
        uv = uv[idx]

        uu, vv = uv[:, 0].astype(int), uv[:, 1].astype(int)
        depth = np.zeros((H, W), dtype=np.float32)
        depth[vv, uu] = z
        
        depth_img = depth_scale_to_integer_factor * depth

        out_name = Path(str(im_data.name)).stem
        depth_path = output_dir / out_name

        save_depth(
            depth=depth_img, depth_path=depth_path, scale_factor=1, verbose=False
        )

        image_id_to_depth_path[im_id] = depth_path
        if include_depth_debug:
            assert (
                input_images_dir is not None
            ), "Need explicit input_images_dir for debug images"
            assert input_images_dir.exists(), input_images_dir

            depth_flat = depth.flatten()[:, None]
            overlay = (
                255.0
                * apply_depth_colormap(torch.from_numpy(depth_flat)).numpy()
            )
            overlay = overlay.reshape([H, W, 3])
            input_image_path = input_images_dir / im_data.name
            input_image = cv2.imread(str(input_image_path))  # type: ignore
            # print(f'input_image.shape: {input_image.shape}, overlay.shape: {overlay.shape}')
            # BUG: why is input image not == overlay image shape?
            if input_image.shape[:2] != overlay.shape[:2]:
                print("images are not the right size!")
                quit()
            debug = 0.3 * input_image + 0.7 + overlay
            out_name = out_name + ".debug.jpg"
            output_path = output_dir / "debug_depth" / out_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), debug.astype(np.uint8))  # type: ignore

    return image_id_to_depth_path

Colormaps = Literal["default", "turbo", "viridis", "magma", "inferno", "cividis", "gray", "pca"]

@dataclass(frozen=True)
class ColormapOptions:
    """Options for colormap"""

    colormap: Colormaps = "default"
    """ The colormap to use """
    normalize: bool = False
    """ Whether to normalize the input tensor image """
    colormap_min: float = 0
    """ Minimum value for the output colormap """
    colormap_max: float = 1
    """ Maximum value for the output colormap """
    invert: bool = False
    """ Whether to invert the output colormap """

def apply_colormap(
    image,
    colormap_options=ColormapOptions(),
    eps=1e-9,
):
    """
    Applies a colormap to a tensor image.
    If single channel, applies a colormap to the image.
    If 3 channel, treats the channels as RGB.
    If more than 3 channel, applies a PCA reduction on the dimensions to 3 channels

    Args:
        image: Input tensor image.
        eps: Epsilon value for numerical stability.

    Returns:
        Tensor with the colormap applied.
    """

    # default for rgb images
    if image.shape[-1] == 3:
        return image

    # rendering depth outputs
    if image.shape[-1] == 1 and torch.is_floating_point(image):
        output = image
        if colormap_options.normalize:
            output = output - torch.min(output)
            output = output / (torch.max(output) + eps)
        output = (
            output * (colormap_options.colormap_max - colormap_options.colormap_min) + colormap_options.colormap_min
        )
        output = torch.clip(output, 0, 1)
        if colormap_options.invert:
            output = 1 - output
        return apply_float_colormap(output, colormap=colormap_options.colormap)

    # rendering boolean outputs
    if image.dtype == torch.bool:
        return apply_boolean_colormap(image)

    if image.shape[-1] > 3:
        return apply_pca_colormap(image)

    raise NotImplementedError


def apply_boolean_colormap(
    image,
    true_color,
    false_color,
):
    """Converts a depth image to color for easier analysis.

    Args:
        image: Boolean image.
        true_color: Color to use for True.
        false_color: Color to use for False.

    Returns:
        Colored boolean image
    """

    colored_image = torch.ones(image.shape[:-1] + (3,))
    colored_image[image[..., 0], :] = true_color
    colored_image[~image[..., 0], :] = false_color
    return colored_image


def apply_float_colormap(image, colormap="viridis"):
    """Convert single channel to a color image.

    Args:
        image: Single channel image.
        colormap: Colormap for image.

    Returns:
        Tensor: Colored image with colors in [0, 1]
    """
    if colormap == "default":
        colormap = "turbo"

    image = torch.nan_to_num(image, 0)
    if colormap == "gray":
        return image.repeat(1, 1, 3)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return torch.tensor(matplotlib.colormaps[colormap].colors, device=image.device)[image_long[..., 0]]

def apply_depth_colormap(
    depth,
    accumulation=None,
    near_plane=None,
    far_plane=None,
    colormap_options= ColormapOptions()):
    """Converts a depth image to color for easier analysis.

    Args:
        depth: Depth image.
        accumulation: Ray accumulation used for masking vis.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.
        colormap: Colormap to apply.

    Returns:
        Colored depth image with colors in [0, 1]
    """

    near_plane = near_plane if near_plane is not None else float(torch.min(depth))
    far_plane = far_plane if far_plane is not None else float(torch.max(depth))

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)
    # depth = torch.nan_to_num(depth, nan=0.0) # TODO(ethan): remove this

    colored_image = apply_colormap(depth, colormap_options=colormap_options)

    if accumulation is not None:
        colored_image = colored_image * accumulation + (1 - accumulation)

    return colored_image


def apply_pca_colormap(
    image, pca_mat=None, ignore_zeros=True
):
    """Convert feature image to 3-channel RGB via PCA. The first three principle
    components are used for the color channels, with outlier rejection per-channel

    Args:
        image: image of arbitrary vectors
        pca_mat: an optional argument of the PCA matrix, shape (dim, 3)
        ignore_zeros: whether to ignore zero values in the input image (they won't affect the PCA computation)

    Returns:
        Tensor: Colored image
    """
    original_shape = image.shape
    image = image.view(-1, image.shape[-1])
    if ignore_zeros:
        valids = (image.abs().amax(dim=-1)) > 0
    else:
        valids = torch.ones(image.shape[0], dtype=torch.bool)

    if pca_mat is None:
        _, _, pca_mat = torch.pca_lowrank(image[valids, :], q=3, niter=20)
    assert pca_mat is not None
    image = torch.matmul(image, pca_mat[..., :3])
    d = torch.abs(image[valids, :] - torch.median(image[valids, :], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    m = 2.0  # this is a hyperparam controlling how many std dev outside for outliers
    rins = image[valids, :][s[:, 0] < m, 0]
    gins = image[valids, :][s[:, 1] < m, 1]
    bins = image[valids, :][s[:, 2] < m, 2]

    image[valids, 0] -= rins.min()
    image[valids, 1] -= gins.min()
    image[valids, 2] -= bins.min()

    image[valids, 0] /= rins.max() - rins.min()
    image[valids, 1] /= gins.max() - gins.min()
    image[valids, 2] /= bins.max() - bins.min()

    image = torch.clamp(image, 0, 1)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return image.view(*original_shape[:-1], 3)


if __name__ == "__main__":
    tyro.cli(ColmapToAlignedMonoDepths).main()
