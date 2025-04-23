import os
import torch
import cv2
import glob
import numpy as np
import argparse

from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

def read_depth_npy(path):
    file_path = path

    # 使用np.load读取.npy文件
    array = np.load(file_path)
    return array


def prepare_gt_depth(benchmark, root_path, output_path):

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.cuda.set_device(0)
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }   
    encoder = 'vitl' # or 'vits', 'vitb'

    # data 
    if benchmark=="DTU":
        encoder = 'vitl' # or 'vits', 'vitb'
        # scenes = ["scan8"] # "scan30",  "scan21",
        scenes = ["scan30", "scan21", "scan34", "scan41", "scan45",  "scan82", "scan103", "scan38", "scan55", "scan40", "scan63", "scan31", "scan8", "scan110", "scan114"]
    elif benchmark=="LLFF":
        encoder = 'vitl' # or 'vits', 'vitb'
        scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
    elif benchmark=="Blender":
        encoder = 'vitl'
        scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
    
    dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20 # 20 for indoor model, 80 for outdoor model

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(f'/media/pc/D/zzt/depth_3DGS/dpt/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    model.to(DEVICE).eval()

    # data_dir
    for dataset_id in scenes:
        if args.root_path[-1]!="/":
            root_path = args.root_path+'/'
        else:
            root_path = args.root_path
        
        # output_path = root_path
        if args.benchmark=="DTU":
            root_path_1 = root_path+dataset_id+'/input/*3_r5000*'
            image_paths_1 = sorted(glob.glob(root_path_1))
            print(f'image_paths_1:{root_path_1}')
            image_path_pkg = [image_paths_1]
            downsampling = 4

        elif args.benchmark=="LLFF":
            root_path_1 = root_path+dataset_id+'/images/*.JPG'
            root_path_2 = root_path+dataset_id+'/images/*.jpg'
            image_paths_1 = sorted(glob.glob(root_path_1))
            image_paths_2 = sorted(glob.glob(root_path_2))
            image_path_pkg = [image_paths_1, image_paths_2]
            # root_path = root_path+'/*png'
            downsampling = 8
        elif args.benchmark=="Blender":
            root_path_1 = root_path+dataset_id+'/train/*png'
            root_path_2 = root_path+dataset_id+'/test/*png'
            image_paths_1 = sorted(glob.glob(root_path_1))
            image_paths_2 = sorted(glob.glob(root_path_2))
            image_path_pkg = [image_paths_1, image_paths_2]
            downsampling = 2
            output_path_1 = os.path.join('/'.join(root_path_1.split('/')[:-1]), 'depth_maps_anything')
            output_path_2 = os.path.join('/'.join(root_path_2.split('/')[:-1]), 'depth_maps_anything')
            output_path_pkg = [output_path_1, output_path_2]
            
            output_path_npy_1 = os.path.join('/'.join(root_path_1.split('/')[:-1]), 'depth_npy_anything')
            output_path_npy_2 = os.path.join('/'.join(root_path_2.split('/')[:-1]), 'depth_npy_anything')
            output_path_npy_pkg = [output_path_npy_1, output_path_npy_2]
            
            for output_path in output_path_pkg:
                if not os.path.exists(output_path): 
                    os.makedirs(output_path, exist_ok=True)
            for output_npy_path in output_path_npy_pkg:
                if not os.path.exists(output_npy_path):
                    os.makedirs(output_npy_path, exist_ok=True)
                    
        if args.benchmark=="DTU" or args.benchmark=="LLFF":
            output_path = os.path.join(root_path+dataset_id, 'depth_maps_anything')
            output_path_npy = os.path.join(root_path+dataset_id, 'depth_npy_anything')

            if not os.path.exists(output_path): 
                os.makedirs(output_path, exist_ok=True)
            if not os.path.exists(output_path_npy):
                os.makedirs(output_path_npy, exist_ok=True)
            output_path_pkg = None
                
        print(output_path_pkg, output_path_npy_pkg)
        for image_paths, output_paths, output_npy_paths in zip(image_path_pkg, output_path_pkg, output_path_npy_pkg):
            for k in range(len(image_paths)):
                filename = image_paths[k]
                # print(f'Progress {k+1}/{len(image_path_pkg)}: {filename}')
                
                raw_image = cv2.imread(filename)
                # print(raw_image.shape)
                if args.benchmark=="LLFF":
                    # input_size=(int(raw_image.shape[1]/downsampling), int(raw_image.shape[0]/downsampling))
                    input_size=((raw_image.shape[0]//downsampling), (raw_image.shape[1]//downsampling))
                elif args.benchmark=="DTU":
                    # input_size=(int(raw_image.shape[0]/downsampling), int(raw_image.shape[1]/downsampling))
                    # print(raw_image.shape[0]/downsampling, raw_image.shape[1]/downsampling)
                    input_size=(298, 398)
                elif args.benchmark=="Blender":
                    input_size=((raw_image.shape[0]//downsampling), (raw_image.shape[1]//downsampling))
                
                depth = model.infer_image(raw_image, input_size)
                if args.benchmark=="DTU" or args.benchmark=="LLFF":
                    np.save(os.path.join(output_npy_paths, os.path.splitext(os.path.basename(filename))[0] + '.npy'), depth)
                elif args.benchmark=="Blender":
                    name = 'depth_'+filename.split('/')[-1]
                    output_file_name = os.path.join(output_npy_paths, name.split('.')[0])
                    np.save(os.path.join(output_npy_paths, os.path.splitext(os.path.basename(filename))[0] + '.npy'), depth)

                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8) 
                if args.benchmark=="Blender":
                    name = 'depth_'+filename.split('/')[-1]
                    output_file_name = os.path.join(output_paths, name.split('.')[0])
                    cv2.imwrite(output_file_name + '.png', depth)
                else:
                    cv2.imwrite(os.path.join(output_path, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)


def prepare_depth(root_path):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.cuda.set_device(0)
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }  
    encoder = 'vitl'
    dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20 # 20 for indoor model, 80 for outdoor model

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(f'./dpt/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    model.to(DEVICE).eval()

    output_path = os.path.join(root_path, 'depth_maps_anydepth')
    output_path_npy = os.path.join(root_path, 'depth_npy_anything')

    if not os.path.exists(output_path): 
        os.makedirs(output_path, exist_ok=True)
    if not os.path.exists(output_path_npy):
        os.makedirs(output_path_npy, exist_ok=True)
    print(f'output_path_npy,{output_path_npy}')
    root_path_1 = root_path+'/img/*.jpg'
    image_paths_1 = sorted(glob.glob(root_path_1))
    image_path_pkg = [image_paths_1]
    for image_paths in image_path_pkg:
            for k in range(len(image_paths)):
                filename = image_paths[k]

                raw_image = cv2.imread(filename)

                depth = model.infer_image(raw_image, (raw_image.shape[0], raw_image.shape[1]))

                np.save(os.path.join(output_path_npy, os.path.splitext(os.path.basename(filename))[0] + '.npy'), depth)

                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8) 
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)

                cv2.imwrite(os.path.join(output_path, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--benchmark', type=str) 
    parser.add_argument('-r', '--root_path', type=str)
    parser.add_argument('-o', '--output_path', type=str)
    args = parser.parse_args()

    prepare_gt_depth(args.benchmark, args.root_path, args.output_path)

