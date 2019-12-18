# #!/bin/bash -l

experiment_name=LWS-3nets
repo_name=NN-augmented-wave-sim

path_script=$HOME/$repo_name/src
path_data=$HOME/$repo_name/vel-model
path_model=$HOME/$repo_name/outputs
mkdir -p $path_model

CUDA_VISIBLE_DEVICES=0 python $path_script/main.py --experiment_dir $experiment_name --phase train \
--same_model_training 1 --correction_num 3 --training_fraction 2 --netEpoch 10 --epoch 500 --epoch_step 100 \
--save_freq 1000  --print_freq 50 --checkpoint_dir $path_model/checkpoint --sample_dir $path_model/sample \
--log_dir $path_model/log --data_path $path_data
