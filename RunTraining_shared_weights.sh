# #!/bin/bash -l

experiment_name=LWS_sharedweights_3corr
repo_name=NN-augmented-wave-sim

path_script=$HOME/$repo_name/src
path_data=$HOME/$repo_name/vel-model
path_model=$HOME/$repo_name/outputs/$experiment_name
mkdir $path_model

CUDA_VISIBLE_DEVICES=0 python $path_script/main_shared_weights.py --experiment_dir $experiment_name --phase train \
--same_model_training 1 --virtSteps 3 --correction_num 1 --training_fraction 2 --epoch 500 --epoch_step 100 \
--save_freq 1000  --print_freq 50 --checkpoint_dir $path_model/checkpoint --sample_dir $path_model/sample \
--log_dir $path_model/log --data_path $path_data
