#!/bin/bash

# This script can optionally manage the main config name via command line passing to Hydra
SCRIPT_PATH="slt.py"

task="dihedral"

# Whether to log particular things
llc="False"
ed="True"

run_idx=1

# Hyperparams
lr=1e-4
its=14000
layers=3
seq_len=25

for i in {1..1}
do
    HYDRA_FULL_ERROR=1 python $SCRIPT_PATH -m num_training_iter=$its lr=$lr task_config=$task seq_len=$seq_len n_layers=$layers run_idx=$run_idx hydra.job.chdir=True

done