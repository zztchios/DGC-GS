import argparse
import numpy as np
import os
import struct
from PIL import Image
import warnings
import os
import matplotlib.pyplot as plt
import glob

warnings.filterwarnings('ignore') # 屏蔽nan与min_depth比较时产生的警告

camnum = 49
fB = 32504
min_depth_percentile = 2
max_depth_percentile = 98

benchmark = "LLFF" # or "DTU"

if benchmark=="DTU":
    scenes = ["scan30", "scan34", "scan41", "scan45",  "scan82", "scan103", "scan38", "scan21", "scan55", "scan40", "scan63", "scan31", "scan8", "scan110", "scan114"]
elif benchmark=="LLFF":
    scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]

depthmapsdir = '/your/depth/map/depth/'
outputdir = '/your/depth/map/depth/depth_colmap/'


def read_path(root_path, benchmark, scenes):

    for dataset_id in scenes:
        if root_path[-1]!="/":
            root_path = root_path+'/'
        else:
            root_path = root_path

        # output_path = root_path
        if benchmark=="DTU":
            root_path_1 = root_path+dataset_id+'/images/*3_r5000*'
            image_paths_1 = sorted(glob.glob(root_path_1))

        elif benchmark=="LLFF":
            root_path_1 = root_path+dataset_id+'/images/*.JPG'
            root_path_2 = root_path+dataset_id+'/images/*.jpg'
            image_paths_1 = sorted(glob.glob(root_path_1))
            image_paths_2 = sorted(glob.glob(root_path_2))
            image_path_pkg = [image_paths_1, image_paths_2]
            # root_path = root_path+'/*png'
            downsampling = 8

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
    # Read depth and normal maps corresponding to the same image.
    if not os.path.exists(depth_map):
        raise FileNotFoundError("file not found: {}".format(depth_map))

    depth_map = read_array(depth_map)
    # print(depth_map.min(), depth_map.max())
    min_depth, max_depth = np.percentile(
         depth_map[depth_map>0], [min_depth_percentile, max_depth_percentile])

    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth


    save_npy_path = os.path.join(os.path.dirname(os.path.dirname(depthdir)), 'depth_colmap_npy', 
                         f"rect_{i:03d}_3_r5000" + '.npy')
    # np.save(save_npy_path, depth_map)

    depth_npy = depth_map
    depth_map = depth_map.astype(np.uint8)
    
    image = Image.fromarray(depth_map).convert('L')
    image = image.resize((398, 298), Image.ANTIALIAS)

    print(save_npy_path, os.path.isdir(save_npy_path))
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(depthdir)), 'depth_colmap_npy'), exist_ok=True)
    np.save(save_npy_path, depth_npy)

def read_depth_npy(path):
    file_path = path

    # 使用np.load读取.npy文件
    array = np.load(file_path)
    
    return array

for j in range(1, camnum+1):
	binjdir = depthmapsdir + f"rect_{j:03d}_3_r5000"  + '.png.' + 'geometric' + '.bin'
	
	# binjdir = depthmapsdir + f"rect_{j:03d}_3_r5000"  + '.png.' + 'photometric' + '.bin'
	print(j)
	binjdir = depthmapsdir + str(j) + '.png.' + 'photometric' + '.bin'
	if os.path.exists(binjdir):
		bin2depth(j, binjdir, outputdir)
        
       
