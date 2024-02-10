#!/bin/bash

# This script can optionally manage the main config name via command line passing to Hydra
SCRIPT_PATH="run.py"

task="adder,abab"
task="alternating,cyclic"
task="dihedral,flipflop"
task="gridworld,parity"
task="quaternion,permutation_reset,symmetric"

# Hyperparams
lr="0.01"
final_lr="0.0005"
its="10000"
eval_f="100"
layers="3"
seq_len="50"

# RLCT
sampler="SGLD"
chains=10
ed_eval_f="10"
rlct_lr="3e-7"

## To put straight into VSCode:
# HYDRA_FULL_ERROR=1 python scripts/run.py optimizer_config.default_lr=0.001 hydra.job.chdir=True dataset_type=adder task_config=adder

for i in {1..1}
do
    HYDRA_FULL_ERROR=1 python $SCRIPT_PATH -m optimizer_config.default_lr=$lr optimizer_config.final_lr=$final_lr num_training_iter=$its eval_frequency=$eval_f task_config=$task task_config.length=$seq_len nano_gpt_config.n_layers=$layers rlct_config.sampling_method=$sampler rlct_config.num_chains=$chains rlct_config.ed_config.eval_frequency=$ed_eval_f rlct_config.sgld_kwargs.lr=$rlct_lr rlct_config.sgnht_kwargs.lr=$rlct_lr hydra.job.chdir=True
done