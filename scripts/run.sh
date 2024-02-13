#!/bin/bash

# This script can optionally manage the main config name via command line passing to Hydra
SCRIPT_PATH="run.py"

# task="adder,cyclic,symmetric"
# task="dihedral,permutation_reset"
# task="abab,flipflop,parity"
# task="symmetric,quaternion,gridworld"

task="quaternion,permutation_reset,symmetric,cyclic"

# Whether to log particular things
llc_train="False"
ed_train="True"
llc_cp="False"
ed_cp="False"

# Hyperparams
# lrs=(1e-3 5e-4 1e-4 1e-5)
# final_lrs=(5e-4 1e-4 5e-5 5e-6)
lrs=(1e-3)
final_lrs=(5e-4)
its=14000
eval_f=100
layers=4
seq_len=25
bs=32
patience=3

# RLCT
sampler="SGLD"
chains=10
ed_eval_f=10
# rlct_lr="3e-8,3e-7,3e-6,3e-5"
# elasticity="1,10"
rlct_lr="3e-7"
elasticity=1
distill="True"
sigma=0.5

## To put straight into VSCode:
# HYDRA_FULL_ERROR=1 python scripts/run.py optimizer_config.default_lr=0.001 hydra.job.chdir=True dataset_type=adder task_config=adder

for i in {1..1}
do
    for i in "${!lrs[@]}"
    do
        HYDRA_FULL_ERROR=1 python $SCRIPT_PATH -m optimizer_config.default_lr=${lrs[$i]} optimizer_config.final_lr=${final_lrs[$i]} dataloader_config.train_bs=$bs num_training_iter=$its eval_frequency=$eval_f task_config=$task ++task_config.length=$seq_len nano_gpt_config.n_layers=$layers early_stop_patience=$patience rlct_config.sampling_method=$sampler rlct_config.num_chains=$chains rlct_config.ed_config.eval_frequency=$ed_eval_f rlct_config.sgld_kwargs.lr=$rlct_lr rlct_config.use_distill_loss=$distill rlct_config.sgld_kwargs.elasticity=$elasticity rlct_config.sgld_kwargs.noise_level=$sigma llc_train=$llc_train ed_train=$ed_train hydra.job.chdir=True
    done
done