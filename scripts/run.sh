#!/bin/bash

# This script can optionally manage the main config name via command line passing to Hydra
SCRIPT_PATH="run.py"

task="adder"
# task="dihedral,permutation_reset,cyclic,symmetric"
# task="abab,flipflop,parity"
# task="quaternion,gridworld,alternating"

# Whether to log particular things
llc_train="True"
ed_train="False"
llc_cp="False"
ed_cp="False"

# Hyperparams
lrs=(1e-4)
final_lrs=(5e-5)
its=14000
eval_f=100
layers=4
seq_len=25
bs=32
patience=4

# RLCT
sampler="SGLD"
chains=10
ed_eval_f=10
rlct_lr="1e-8,5e-8,1e-7,5e-7,1e-6"
elasticity="1,10,50,100"
sigma="0.25,1.0"
# rlct_lr=3e-7
# sigma=0.1
# elasticity=1
distill="True"
box_size="0.1,0.5,1,3,5,null"

## To put straight into VSCode:
# HYDRA_FULL_ERROR=1 python scripts/run.py optimizer_config.default_lr=0.001 hydra.job.chdir=True dataset_type=adder task_config=adder

for i in {1..1}
do
    for i in "${!lrs[@]}"
    do
        HYDRA_FULL_ERROR=1 python $SCRIPT_PATH -m optimizer_config.default_lr=${lrs[$i]} optimizer_config.final_lr=${final_lrs[$i]} dataloader_config.train_bs=$bs num_training_iter=$its eval_frequency=$eval_f task_config=$task ++task_config.length=$seq_len nano_gpt_config.n_layers=$layers early_stop_patience=$patience rlct_config.sampling_method=$sampler rlct_config.num_chains=$chains rlct_config.ed_config.eval_frequency=$ed_eval_f rlct_config.sgld_kwargs.lr=$rlct_lr rlct_config.use_distill_loss=$distill rlct_config.sgld_kwargs.elasticity=$elasticity rlct_config.sgld_kwargs.noise_level=$sigma llc_train=$llc_train ed_train=$ed_train rlct_config.sgld_kwargs.bounding_box_size=$box_size hydra.job.chdir=True
    done
done