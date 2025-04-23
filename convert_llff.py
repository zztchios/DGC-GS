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
import logging
from argparse import ArgumentParser
import shutil
from PIL import Image
# 114 63 45 55 38 34 21
# 8 30 41 82 103
# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
# parser.add_argument("--camera", default="PINHOLE", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--resize_file", "-r", action="store_true")
parser.add_argument("--step", "-t", action="store_true")


args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 0 if args.no_gpu else 1

def resize_file(dirfile):
    dir_file =  os.path.join(dirfile, 'input')
    target_width = 398
    target_height = 298
    os.makedirs(os.path.join(dirfile, "input_resize"), exist_ok=True)
    for filename in os.listdir(dir_file):
        file_save_path = os.path.join(dirfile, "input_resize", os.path.basename(filename))
        print("save path: ", file_save_path)
        with Image.open(os.path.join(dir_file, filename)) as img:
            # 调整图像大小
            resized_img = img.resize((target_width, target_height), Image.ANTIALIAS)
            resized_img.save(file_save_path)

if args.resize_file:
    resize_file(args.source_path)

if not args.skip_matching and not args.step:
    os.makedirs(args.source_path + "/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/database.db \
        --image_path " + args.source_path + "/images\
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/database.db \
        --image_path "  + args.source_path + "/images \
        --output_path "  + args.source_path + "/sparse")
        # --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)
    
    


# img_undist_cmd = (colmap_command + " image_undistorter \
#     --image_path " + args.source_path + "/input \
#     --input_path " + args.source_path + "/distorted/sparse/0 \
#     --min_scale 1 \
#     --output_path " + args.source_path + "\
#     --output_type COLMAP")
# exit_code = os.system(img_undist_cmd)



''' '''
# os.makedirs(args.source_path + "/dense/sparse/0", exist_ok=True)
## Image undistortion
# We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/images \
    --input_path " + args.source_path + "/sparse/0 \
    --output_path " + args.source_path + "/dense\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"undist failed with code {exit_code}. Exiting.")
    exit(exit_code)   

files = os.listdir(args.source_path + "/dense/sparse")
# os.makedirs(args.source_path + "/dense/sparse/1", exist_ok=True)
# Copy each file from the source directory to the destination directory
# for file in files:
#     if file == '0':
#         continue
#     source_file = os.path.join(args.source_path, "dense/sparse", file)
#     destination_file = os.path.join(args.source_path, "dense/sparse", "0", file)
#     shutil.move(source_file, destination_file)

# os.makedirs(args.source_path + "/dense", exist_ok=True)
# os.makedirs(args.source_path + "/dense/sparse", exist_ok=True)

img_patch_match_cmd = (colmap_command + " patch_match_stereo \
    --workspace_path " + args.source_path + "/dense \
    --workspace_format COLMAP" + " \
    --PatchMatchStereo.geom_consistency true")

exit_code = os.system(img_patch_match_cmd)
if exit_code != 0:
    logging.error(f"Patch_match failed with code {exit_code}. Exiting.")
    exit(exit_code)

img_stereo_fusion_cmd = (colmap_command + " stereo_fusion \
    --workspace_path " + args.source_path + "/dense \
    --workspace_format COLMAP" + " \
    --input_type geometric" +"\
    --output_path " + args.source_path + "/dense/fused.ply")

exit_code = os.system(img_stereo_fusion_cmd)
if exit_code != 0:
    logging.error(f"Stereo_fusion failed with code {exit_code}. Exiting.")
    exit(exit_code)

# colmap poisson_mesher --input_path dense/fused.ply --output_path dense/meshed-poisson.ply
img_poisson_mesher_cmd = (colmap_command + " poisson_mesher \
    --input_path " + args.source_path + "/dense/fused.ply \
    --output_path " + args.source_path + "/dense/meshed-poisson.ply")

exit_code = os.system(img_poisson_mesher_cmd)
if exit_code != 0:
    logging.error(f"Poisson_mesher failed with code {exit_code}. Exiting.")
    exit(exit_code)

img_delaunay_mesher_cmd = (colmap_command + " delaunay_mesher \
    --input_path " + args.source_path + "/dense\
    --output_path " + args.source_path + "/dense/meshed-delaunay.ply")

exit_code = os.system(img_delaunay_mesher_cmd)
if exit_code != 0:
    logging.error(f"Poisson_mesher failed with code {exit_code}. Exiting.")
    exit(exit_code)

if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")


