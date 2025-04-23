import cv2
import torch

# import matplotlib.pyplot as plt
import utils_io

import numpy as np
import os
import argparse
import glob

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root_path', type=str)
args = parser.parse_args()


repo = "/home/pc/.cache/torch/hub/intel-isl_MiDaS"

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
# model_type = "DPT_BEiT_L_384"

# midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas = torch.hub.load(repo, model_type, path='/home/pc/.cache/torch/hub/checkpoints/dpt_large_384.pt', source='local')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
midas_transforms = torch.hub.load(repo, "transforms", source='local')

if "DPT" in model_type:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


for dataset_id in ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]:
    if args.root_path[-1]!="/":
        root_path = args.root_path+'/'
    else:
        root_path = args.root_path

    # output_path = root_path

    root_path_1 = root_path+dataset_id+'/train/*png'
    root_path_2 = root_path+dataset_id+'/test/*png'
    image_paths_1 = sorted(glob.glob(root_path_1))
    image_paths_2 = sorted(glob.glob(root_path_2))
    image_path_pkg = [image_paths_1, image_paths_2]

    output_path_1 = os.path.join('/'.join(root_path_1.split('/')[:-1]), 'depth_maps')
    output_path_2 = os.path.join('/'.join(root_path_2.split('/')[:-1]), 'depth_maps')
    output_path_pkg = [output_path_1, output_path_2]

    print('image_paths:', image_path_pkg)

    downsampling = 2
    for output_path in output_path_pkg:
        if not os.path.exists(output_path): 
            os.makedirs(output_path, exist_ok=True)
    for image_paths, output_path in zip(image_path_pkg, output_path_pkg):
        for k in range(len(image_paths)):
            filename = image_paths[k]
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, (img.shape[1] // 8, img.shape[0] // 8), interpolation=cv2.INTER_CUBIC)
            print('k, img.shape:', k, img.shape) #(1213, 1546, 3)
            h, w = img.shape[:2]
            input_batch = transform(img).to(device)

            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(h//downsampling, w//downsampling),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            output = prediction.cpu().numpy()
            name = 'depth_'+filename.split('/')[-1]
            print('######### output_path and name:', output_path,  name)
            output_file_name = os.path.join(output_path, name.split('.')[0])
            # utils.io.write_depth(output_file_name.split('.')[0], output, bits=2)
            utils_io.write_depth(output_file_name, output, bits=2)