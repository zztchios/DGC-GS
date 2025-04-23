#!/bin/bash
input=$1    # ./data/nerf_llff_data/
output=$2    # './data/llff/'
# scan=("scan8" "scan21" "scan30" "scan31" "scan34" "scan38" "scan40" "scan41" "scan45" "scan55" "scan63" "scan82" "scan103" "scan110" "scan114")
# scene=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
# eps=(0.002 0.006 0.006 0.006 0.006 0.006 0.06 0.006)
scene=("horns")
eps=(0.01)
for i in "${!scene[@]}"; do
    scan=${scene[$i]}
    epsilon=${eps[$i]}
    
    python colmap_depth.py -i /media/pc/D/zzt/DNGaussian/DNGaussian0926/data/nerf_llff_data/ \
            -s $scan -o ./data/llff/ -d LLFF
    # echo "depth_scale_align_llff $scan"
    # python depth_scale_align_llff_patch.py -s $scan -i $input -o $output --eps $epsilon
done