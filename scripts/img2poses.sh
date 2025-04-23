#!/bin/bash
scan=("scan8" "scan21" "scan30" "scan31" "scan34" "scan38" "scan40" "scan41" "scan45" "scan55" "scan63" "scan82" "scan103" "scan110" "scan114")
for scan in "${scan[@]}"; do
    echo "/media/pc/D/zzt/CoR-GS/data/DTU_new/$scan/"
    python image2poses.py --working_dir /media/pc/D/zzt/CoR-GS/data/DTU_new/$scan/
done