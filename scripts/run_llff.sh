dataset=$1 
workspace=$2 
export CUDA_VISIBLE_DEVICES=$3
config=$4
iter=$5



python train_llff.py  --source_path $dataset --model_path $workspace --lambda_dssim 0.2 \
                           --eval  --n_views 3 --sample_pseudo_interval 1 \
                           --n_sparse 3 -r 8 --iterations $iter \
                           --near 10 \
                           --densify_grad_threshold 0.0013 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01 \
                           --split_opacity_thresh 0.1 --error_tolerance 0.00025 \
                           --scaling_lr 0.003 \
                           --shape_pena 0.002 --opa_pena 0.001 \
                           --save_warp \
                           --config $config  \
                           --mvs_pcd
                        
python render.py --source_path $dataset --model_path $workspace -r 8 --near 10 --render --iteration $iter 
python spiral.py --source_path $dataset --model_path $workspace -r 8 --near 10 --render

python metrics.py  --model_path $workspace