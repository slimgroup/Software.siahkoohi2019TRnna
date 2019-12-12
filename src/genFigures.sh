#!/bin/bash -l

repo_name=NN-augmented-wave-sim

experiment_name=LWS-3nets
result_path=$HOME/$repo_name/outputs/$experiment_name/sample

path_script=$HOME/$repo_name/src
savepath=$HOME/$repo_name/outputs/$experiment_name/figs

python show_prediction.py --hdf5path $result_path --savepath $savepath