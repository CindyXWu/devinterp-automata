#!/bin/bash

# This script can optionally manage the main config name via command line passing to Hydra
SCRIPT_PATH="run.py"

task="dihedral"
# task="dihedral,permutation_reset,cyclic,symmetric"
# task="abab,flipflop,parity"
# task="quaternion,gridworld,alternating"

# Whether to log particular things
llc_train="True"
ed_train="True"
llc_cp="False"
ed_cp="False"

# Hyperparams
lr=1e-4
final_lr=5e-5
its=14000
eval_f=100
layers=3
seq_len=25
bs=64
patience=8

# RLCT
sampler="SGLD_MA"
chains=10
ed_eval_f=10
distill="True"
rlct_lrs=(1e-10 1e-9)
weight_decays=(1e-9 1e-9)
# rlct_lrs=(1e-9)
# weight_decays=(1e-9)
sigma=1.0
box_size=10
elasticity=10

for i in {1..1}
do
    for i in "${!rlct_lrs[@]}"
    do
        HYDRA_FULL_ERROR=1 python $SCRIPT_PATH -m optimizer_config.default_lr=$lr optimizer_config.final_lr=$final_lr dataloader_config.train_bs=$bs num_training_iter=$its eval_frequency=$eval_f task_config=$task ++task_config.length=$seq_len nano_gpt_config.n_layers=$layers early_stop_patience=$patience rlct_config.sampling_method=$sampler rlct_config.num_chains=$chains rlct_config.ed_config.eval_frequency=$ed_eval_f rlct_config.sgld_kwargs.lr=${rlct_lrs[$i]} rlct_config.sgld_kwargs.weight_decay=${weight_decays[$i]} rlct_config.use_distill_loss=$distill rlct_config.sgld_kwargs.elasticity=$elasticity rlct_config.sgld_kwargs.noise_level=$sigma llc_train=$llc_train ed_train=$ed_train rlct_config.sgld_kwargs.bounding_box_size=$box_size hydra.job.chdir=True
    done
done