#!/bin/bash

# This script can optionally manage the main config name via command line passing to Hydra
SCRIPT_PATH="slt.py"

task="symmetric"
model=TF_LENS

# Whether to log particular things
llc="False"
ed="True"
osculating="True"
form="True"

run_idx=0
skip_cps=1

# Hyperparams
lr=5e-4
its=100000
truncate_its=25000 # Set to null for auto
layers=3
seq_len=25

# ED
early_sigma=30
late_sigma=50

for i in {1..1}
do
    HYDRA_FULL_ERROR=1 python $SCRIPT_PATH -m num_training_iter=$its model_type=$model llc=$llc ed=$ed form=$form osculating=$osculating lr=$lr task_config=$task seq_len=$seq_len n_layers=$layers run_idx=$run_idx ed_plot_config.smoothing_sigma_early=$early_sigma skip_cps=$skip_cps truncate_its=$truncate_its ed_plot_config.smoothing_sigma_late=$late_sigma hydra.job.chdir=True

done