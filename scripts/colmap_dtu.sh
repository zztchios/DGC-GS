scan_id=$1
min_depth_percentile=$2
max_depth_percentile=$3
input=$4    # ./data/dtu
output=$5    # '/media/pc/D/zzt/CoR-GS/data/DTU_new/'
echo "colmap_depth $scan_id"
python colmap_depth.py -s $scan_id -n $min_depth_percentile -x $max_depth_percentile -i $input -o $input -d DTU
echo "depth_scale_align_dtu_wmask $scan_id"
python depth_scale_align_dtu_wmask.py -s $scan_id -i $input -o $output
echo "depth_scale_align_dtu_womask $scan_id"
python depth_scale_align_dtu_womask.py -s $scan_id -i $input -o $output

