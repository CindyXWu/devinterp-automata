#!/bin/bash

# This script can optionally manage the main config name via command line passing to Hydra
SCRIPT_PATH="slt.py"

task="dihedral"

# Whether to log particular things
llc="False"
ed="True"
form="False"

run_idx=1

# Hyperparams
lr=1e-4
its=14000
layers=3
seq_len=25

# ED
early_sigma=1000
late_sigma=1500
marked_cusp_data="None"
colors="[[255, 0, 0], [0, 255, 0], [0, 0, 255]]"

for i in {1..1}
do
    HYDRA_FULL_ERROR=1 python $SCRIPT_PATH -m num_training_iter=$its llc=$llc ed=$ed form=$form lr=$lr task_config=$task seq_len=$seq_len n_layers=$layers run_idx=$run_idx ed_plot_config.smoothing_sigma_early=$early_sigma ed_plot_config.smoothing_sigma_late=$late_sigma colors=$colors hydra.job.chdir=True

done