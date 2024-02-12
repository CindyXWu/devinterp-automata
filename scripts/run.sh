#!/bin/bash

# This script can optionally manage the main config name via command line passing to Hydra
SCRIPT_PATH="run.py"

# task="adder,abab,alternating,dihedral"
# task="cyclic,flipflop,gridworld,parity"
task="symmetric,quaternion,gridworld"

# Whether to log particular things
llc_train="False"
ed_train="True"

# Hyperparams
lrs=(1e-3 5e-4 1e-4 1e-5)
final_lrs=(5e-4 1e-4 5e-5 5e-6)
its="10000"
eval_f="200"
layers="4"
seq_len="25"

# RLCT
sampler="SGLD"
chains=10
ed_eval_f="10"
# rlct_lr="5e-8,3e-7,3e-6"
# elasticity="1,10,100"
rlct_lr="3e-7"
elasticity="1"
distill="True"

## To put straight into VSCode:
# HYDRA_FULL_ERROR=1 python scripts/run.py optimizer_config.default_lr=0.001 hydra.job.chdir=True dataset_type=adder task_config=adder

for i in {1..1}
do
    for i in "${!lrs[@]}"
    do
        HYDRA_FULL_ERROR=1 python $SCRIPT_PATH -m optimizer_config.default_lr=${lrs[$i]} optimizer_config.final_lr=${final_lrs[$i]} num_training_iter=$its eval_frequency=$eval_f task_config=$task ++task_config.length=$seq_len nano_gpt_config.n_layers=$layers rlct_config.sampling_method=$sampler rlct_config.num_chains=$chains rlct_config.ed_config.eval_frequency=$ed_eval_f rlct_config.sgld_kwargs.lr=$rlct_lr rlct_config.sgnht_kwargs.lr=$rlct_lr rlct_config.use_distill_loss=$distill rlct_config.sgld_kwargs.elasticity=$elasticity llc_train=$llc_train ed_train=$ed_train hydra.job.chdir=True
    done
done