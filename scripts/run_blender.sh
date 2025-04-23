dataset=$1 
workspace=$2 
export CUDA_VISIBLE_DEVICES=$3
config=$4
iter=$5


## for  drums

scenes=("drums")
for scene in "${scenes[@]}"; do
    # 构建dataset和workspace路径
    dataset_scene="${dataset}/${scene}"
    workspace_scene="${workspace}/${scene}"
    python train_blender.py --dataset blender -s $dataset_scene --model_path $workspace_scene -r 2 --eval --n_sparse 8 --rand_pcd --iterations $iter --lambda_dssim 0.6 --white_background \
                --densify_grad_threshold 0.001 --prune_threshold 0.01 --densify_until_iter 6000 --percent_dense 0.01 \
                --densify_from_iter 500 \
                --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_max_steps 1000 --position_lr_start 5000 \
                --test_iterations 1000 2000 3000 4500 6000 --save_iterations 1000 2000 3000 6000 \
                --split_opacity_thresh 0.1 --error_tolerance 0.001 \
                --theta_range_deg 6 3 2 1 -2 -3 -6 \
                --translate_range 0.02 0.04 0.10 0.16 0.10 0.04 0.02 \
                --scaling_lr 0.005 \
                --config $config  \
                --shape_pena 0.000 --opa_pena 0.000 --scale_pena 0.000 \
                --use_grad --use_rep

    python render.py -s $dataset_scene --model_path $workspace_scene -r 2 --render
    python metrics.py --model_path $workspace_scene 
done


scenes=("lego" "ship")

for scene in "${scenes[@]}"; do
    # 构建dataset和workspace路径
    dataset_scene="${dataset}/${scene}"
    workspace_scene="${workspace}/${scene}"  # 可根据需要调整workspace子目录

    python train_blender.py --dataset blender -s $dataset_scene --model_path $workspace_scene -r 2 --eval --save_warp --n_sparse 8 --rand_pcd --iterations $iter --lambda_dssim 0.2 --white_background \
                --densify_grad_threshold 0.0002 --prune_threshold 0.005 --densify_until_iter 6000 --percent_dense 0.01 \
                --densify_from_iter 500 \
                --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_max_steps 1000 --position_lr_start 5000 \
                --test_iterations 1000 2000 3000 4500 6000 --save_iterations 1000 2000 3000 6000 \
                --error_tolerance 0.01 \
                --scaling_lr 0.005 \
                --shape_pena 0.000 --opa_pena 0.000 --scale_pena 0.000 \
                --theta_range_deg 6 3 2 1 -2 -3 -6 \
                --translate_range 0.02 0.04 0.10 0.16 0.10 0.04 0.02 \
                --config $config  \
                --use_SH # --use_grad --use_rep

    python render_sh.py -s $dataset_scene --model_path $workspace_scene -r 2 --render
    python metrics.py --model_path $workspace_scene
done



scenes=("mic")

for scene in "${scenes[@]}"; do
    # 构建dataset和workspace路径
    dataset_scene="${dataset}/${scene}"
    workspace_scene="${workspace}/${scene}"  # 可根据需要调整workspace子目录

    python train_blender.py --dataset blender -s $dataset_scene --model_path $workspace_scene -r 2 --eval --save_warp --n_sparse 8 --rand_pcd --iterations $iter --lambda_dssim 0.2 --white_background \
                --densify_grad_threshold 0.0002 --prune_threshold 0.005 --densify_until_iter 6000 --percent_dense 0.01 \
                --densify_from_iter 500 \
                --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_max_steps 1000 --position_lr_start 5000 \
                --test_iterations 1000 2000 3000 4500 6000 --save_iterations 1000 2000 3000 6000 \
                --error_tolerance 0.01 \
                --scaling_lr 0.005 \
                --shape_pena 0.000 --opa_pena 0.000 --scale_pena 0.000 \
                --theta_range_deg 6 3 2 1 -2 -3 -6 \
                --translate_range 0.02 0.04 0.10 0.16 0.10 0.04 0.02 \
                --config $config  \
                --use_SH

    python render_sh.py -s $dataset_scene --model_path $workspace_scene -r 2 --render
    python metrics.py --model_path $workspace_scene
done



scenes=("hotdog" "lego" "materials" "chair" "ficus")

for scene in "${scenes[@]}"; do
    # 构建dataset和workspace路径
    dataset_scene="${dataset}/${scene}"
    workspace_scene="${workspace}/${scene}"  # 可根据需要调整workspace子目录

    python train_blender.py --dataset blender -s $dataset_scene --model_path $workspace_scene -r 2 --eval --save_warp --n_sparse 8 --rand_pcd --iterations $iter --lambda_dssim 0.2 --white_background \
                --densify_grad_threshold 0.0002 --prune_threshold 0.005 --densify_until_iter 6000 --percent_dense 0.01 \
                --densify_from_iter 500 \
                --position_lr_init 0.00016 --position_lr_final 0.0000016 --position_lr_max_steps 1000 --position_lr_start 5000 \
                --test_iterations 1000 2000 3000 4500 6000 --save_iterations 1000 2000 3000 6000 \
                --error_tolerance 0.01 \
                --scaling_lr 0.005 \
                --shape_pena 0.000 --opa_pena 0.000 --scale_pena 0.000 \
                --theta_range_deg 6 3 2 1 -2 -3 -6 \
                --translate_range 0.02 0.04 0.10 0.10 0.10 0.04 0.02 \
                --config $config  \
                --use_SH --use_grad --use_rep

    python render_sh.py -s $dataset_scene --model_path $workspace_scene -r 2 --render
    python metrics.py --model_path $workspace_scene
done

python metrics_count.py $workspace $iter


