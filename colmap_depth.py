import argparse
import numpy as np
import os
import struct
from PIL import Image
import warnings
import os
import re
import matplotlib.pyplot as plt
from skimage.transform import resize
import glob

warnings.filterwarnings('ignore') # 屏蔽nan与min_depth比较时产生的警告

from argparse import ArgumentParser

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap depth")
parser.add_argument("--min_depth_percentile", "-n", type=int, default=2)
parser.add_argument("--max_depth_percentile", "-x", type=int, default=98)
parser.add_argument("--scan", "-s", required=True, type=str)
parser.add_argument("--input", "-i", required=True, type=str)
parser.add_argument("--output", "-o", required=True, type=str)
parser.add_argument("--dataset", "-d", required=True, type=str)
args = parser.parse_args()

camnum = 49
min_depth_percentile = args.min_depth_percentile
max_depth_percentile = args.max_depth_percentile
# /mnt/e/dataset/DRGaussian/DTU_DR/scan30
# /mnt/e/dataset/nyuv2/NYU_v2_dataset/data/train
scan = args.scan
# depthmapsdir = '/media/pc/D/zzt/DNGaussian/data_download/dtu/' + scan + '/dense/stereo/depth_maps/'
# outputdir = '/media/pc/D/zzt/DNGaussian/data_download/dtu/' + scan + '/depth_colmap/'
    
# depthmapsdir = '/media/pc/D/zzt/DNGaussian/DNGaussian_Dataset/DTU/' + scan + '/dense/stereo/depth_maps/'
depthmapsdir = args.input + scan + '/3_views/dense/stereo/depth_maps/'
outputdir = args.output + scan + '/depth_colmap_new/'

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

def bin2depth(i, depth_map, depthdir):
    # depth_map = '0.png.geometric.bin'
    # print(depthdir)
    # if min_depth_percentile > max_depth_percentile:
    #     raise ValueError("min_depth_percentile should be less than or equal "
    #                      "to the max_depth_perceintile.")

    # Read depth and normal maps corresponding to the same image.
    
    if not os.path.exists(depth_map):
        raise FileNotFoundError("file not found: {}".format(depth_map))
    depth_map = read_array(depth_map)
    os.makedirs(args.output + args.scan, exist_ok=True)
    os.makedirs(depthdir, exist_ok=True)
    
    
    
    min_depth, max_depth = np.percentile(
         depth_map[depth_map>0], [min_depth_percentile, max_depth_percentile])

    
    depth_map[depth_map <= 0] = np.nan # 把0和负数都设置为nan，防止被min_depth取代
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth
    
    # maxdisp = fB / min_depth
    # mindisp = fB / max_depth
    # depth_map = (fB/depth_map - mindisp) * 255 / (maxdisp - mindisp)
    # print(depth_map.min(), depth_map.max())
 
    if args.dataset == "DTU":
        save_npy_path = os.path.join(os.path.dirname(os.path.dirname(depthdir)), 'depth_pho_colmap_npy', 
                         f"rect_{i:03d}_3_r5000" + '.npy')

    elif args.dataset == "LLFF":
        # 分离文件名和扩展名
        name_part, _ = os.path.splitext(i)  # 去掉最后一个扩展名
        name_part = os.path.splitext(name_part)[0]  # 去掉 .geometric
        
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
            
            # 构建新的文件路径
            save_npy_path = os.path.join(os.path.dirname(os.path.dirname(depthdir)), 'depth_pho_colmap_npy', new_filename)

    
    # np.save(save_npy_path, depth_map)
    depth_npy = depth_map
    depth_npy = np.nan_to_num(depth_npy) # nan全都变为0
    
    
    # 执行缩放操作
    if args.dataset=="DTU":
        depth_npy = resize(depth_npy, (298, 398))
    elif args.dataset=="LLFF":
        depth_npy = resize(depth_npy, (3024//8, 4032//8))

    print(f'depth max: {depth_npy.max()}, min: {depth_npy.min()}')
    depth_map = (depth_npy - depth_npy.min()) * 255.0 / (depth_npy.max() - depth_npy.min()) 
    # depth_map = np.nan_to_num(depth_map) # nan全都变为0
    # depth_map = depth_map[np.newaxis, np.newaxis]

    depth_map = depth_map.astype(np.uint8)
    
    image = Image.fromarray(depth_map).convert('L')
    if args.dataset=="DTU":
        image = image.resize((398, 298), resample=Image.Resampling.LANCZOS) # 保证resize为1920*1080
    elif args.dataset=="LLFF":
        image = image.resize((4032//8, 3024//8), resample=Image.Resampling.LANCZOS)
    # print(save_npy_path, depthdir + f"rect_{i:03d}_3_r5000" + '.png')
    # print(os.path.join(os.path.dirname(os.path.dirname(depthdir)), 'depth_pho_colmap_npy'))
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(depthdir)), 'depth_pho_colmap_npy'), exist_ok=True)
    np.save(save_npy_path, depth_npy)
    
    plt_combined_histogram(depth_npy)
    
    # error(save_npy_path)
    if args.dataset=="DTU":
        image.save(depthdir + f"rect_{i:03d}_3_r5000" + '.png')
    elif args.dataset=="LLFF":
         # 分离文件名和扩展名
        name_part, _ = os.path.splitext(i)  # 去掉最后一个扩展名
        name_part = os.path.splitext(name_part)[0]  # 去掉 .geometric
        
        # 检查文件名是否包含 .JPG 或 .jpg
        if '.JPG' in name_part or '.jpg' in name_part:
            # 提取 .JPG 或 .jpg 之前的部分
            base_name = name_part.split('.')[0]
            new_filename = f"{base_name}.png"
            
            # 构建新的文件路径
            save_dep_path = os.path.join(outputdir, new_filename)

        image.save(save_dep_path)

def read_depth_npy(path):
    file_path = path

    # 使用np.load读取.npy文件
    array = np.load(file_path)
    print("array:", array.max(), array.min())
    return array

def error(save_npy_path):
    npy_file = read_depth_npy(save_npy_path)
    # print('/media/pc/D/zzt/depth_3DGS/datasets/dtu/scan40/depth_anything_npy/'+save_npy_path.split("/")[-1])
    npy_file_depth_anything = read_depth_npy('/media/pc/D/zzt/depth_3DGS/data/dtu/scan8/depth_npy/'+save_npy_path.split("/")[-1])
    # print("npy_file", npy_file.size, npy_file_depth_anything.size)
    # print(npy_file - npy_file_depth_anything)
    print(npy_file.shape)
    error_map = npy_file - npy_file_depth_anything
    # print('/media/pc/D/zzt/depth_3DGS/datasets/dtu/scan40/error/'+os.path.splitext(os.path.basename(save_npy_path))[0]+'.png')
    plt.imshow(error_map, cmap='viridis')  # 'viridis' 是一种常用的颜色映射
    plt.colorbar(label='Depth Error')
    plt.title('Depth Error Map')
    plt.imsave('/media/pc/D/zzt/depth_3DGS/datasets/dtu/scan8/error/'+os.path.splitext(os.path.basename(save_npy_path))[0]+'.png', error_map, cmap='viridis')


def gen_mask(depth):
    depth = np.nan_to_num(depth) # nan全都变为0
    # 步骤1：过滤掉所有小于1的值，并保持它们的索引
    valid_indices = depth >= 1
    valid_values = depth[valid_indices]

    # 找到data_2d中的最大值，并设定阈值
    max_value = np.max(depth)
    threshold_value = max_value / 3  # 假设阈值为最大值的三分之一

    # 步骤3：计算有效元素的数量和小于阈值的有效元素数量
    total_valid_elements = valid_values.size
    print(f'total_valid:{total_valid_elements}')
    below_threshold_indices = valid_values < threshold_value
    num_below_threshold = np.sum(below_threshold_indices)
    
    target_percentage = 30
    target_num_below_threshold = max(1, int(total_valid_elements * target_percentage / 100))
    
    if num_below_threshold >= target_num_below_threshold:
        # 如果满足条件，则创建mask
        mask = np.zeros_like(depth, dtype=bool)
        mask[valid_indices] = below_threshold_indices
    else:
        # 步骤2：对剩下的值进行排序（从小到大）
        sorted_indices = np.argsort(valid_values)

        # 计算需要选择多少个最小的元素作为有用的mask
        num_elements_to_select = max(1, int(total_valid_elements * target_percentage / 100))
        print(f'less_mask: {num_elements_to_select}, {depth.min()}, {depth.max()}')
        # 步骤3和4：创建一个全为False的布尔数组，长度与原始数组相同
        mask = np.zeros_like(depth, dtype=bool)

        # 获取原始二维数组中有效值的位置
        valid_positions = np.where(valid_indices)
        print(f'num_elements_to_select:{num_elements_to_select}')
        # 将最小的30%的值的位置设置为True，在原始数组中的对应位置也设置为True
        smallest_percent_indices = sorted_indices[:num_elements_to_select]
        mask_indices = (valid_positions[0][smallest_percent_indices],
                        valid_positions[1][smallest_percent_indices])
        mask[mask_indices] = True
    
    return mask

    
def plt_combined_histogram(data, save_name='combined_histogram.png', dpi=300, format='png'):
    """
    绘制并保存两个维度数据合并后的直方图。

    参数:
    data (numpy.ndarray): 二维数据集，形状为 (n_samples, 2)
    save_path (str): 图像保存路径，默认为 'combined_histogram.png'
    dpi (int): 图像分辨率，默认为 300 DPI
    format (str): 图像格式，默认为 'png'
    """

    # 将两个维度的数据合并成一个一维数组
    combined_data = data.flatten()

    # 创建直方图
    plt.figure(figsize=(10, 6))
    plt.hist(combined_data, bins=100, color='skyblue', edgecolor='black')
    plt.title('Combined Histogram of Both Dimensions')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # 调整布局以避免标签被裁剪
    plt.tight_layout()

    # 保存图像
    plt.savefig(os.path.join(args.output, scan, scan + "_hist.png"), dpi=300, format='png', bbox_inches='tight')

    # 关闭当前图形以释放内存
    plt.close()

file_pattern = os.path.join(depthmapsdir, '*.geometric.bin')
bin_files = glob.glob(file_pattern, recursive=True)

'''  '''

if args.dataset == 'DTU':
    # for j in range(1, camnum+1):
    #     binjdir = bin_files + f"rect_{j:03d}_3_r5000"  + '.png.' + 'geometric' + '.bin'
    for i, file in enumerate(bin_files):
        if os.path.exists(file):
            bin2depth(i, file, outputdir)
elif args.dataset == 'LLFF':
    binjdir = bin_files
    for file in binjdir:
        # 规范化路径，确保没有冗余的分隔符或相对路径部分
        normalized_path = os.path.normpath(file)
        
        # 分离出路径中固定的部分和通配符部分（如果有）
        base_path, wildcard_part = os.path.split(normalized_path)
        
        # 构建正则表达式模式以匹配要移除的目录
        pattern = re.escape(os.sep + '3_views' + os.sep)
        
        # 移除指定的目录层级
        new_base_path = re.sub(pattern, os.sep, base_path)
        
        # 如果原始路径以斜杠结尾或者有通配符部分，则保留这些特性
        if wildcard_part:
            new_path = os.path.join(new_base_path, wildcard_part)
        else:
            new_path = new_base_path
        
        file = os.path.normpath(new_path)
        print(f'file:{file}')
        if os.path.exists(file):
            bin2depth(os.path.basename(file), file, outputdir)
    
    # binjdir = depthmapsdir + f"rect_{j:03d}_3_r5000"  + '.png.' + 'photometric' + '.bin'
    # print(binjdir)
    # binjdir = depthmapsdir + str(j) + '.png.' + 'photometric' + '.bin'
    
         

# print(read_depth_npy('/media/pc/D/zzt/DNGaussian/data/dtu/scan40/depth_colmap_npy/rect_001_3_r5000.npy').shape)
# print(read_depth_npy('/media/pc/D/zzt/DNGaussian/data/dtu/scan40/depth_pho_colmap_npy/rect_001_3_r5000.npy').shape)
# error('/media/pc/D/zzt/depth_3DGS/data/dtu/scan8/depth_colmap_npy/rect_001_3_r5000.npy')
          
       
