#!/bin/bash

# This script can optionally manage the main config name via command line passing to Hydra
SCRIPT_PATH="run.py"

task="symmetric"
# task="dihedral,permutation_reset,cyclic,symmetric"
# task="abab,flipflop,parity"
# task="quaternion,gridworld,alternating"

# Whether to log particular things
llc_train="True"
ed_train="False"
llc_cp="False"
ed_cp="False"

# Hyperparams
its=14000
eval_f=100
layers=3
seq_len=25

# RLCT
sampler="SGLD_MA"
chains=5
ed_eval_f=100
distill="True"
rlct_lrs=(1e-8)
weight_decays=(1e-8)
mh_freq=10
chain_len=5000

for i in {1..1}
do
    for i in "${!rlct_lrs[@]}"
    do
        HYDRA_FULL_ERROR=1 python $SCRIPT_PATH -m num_training_iter=$its eval_frequency=$eval_f task_config=$task ++task_config.length=$seq_len nano_gpt_config.n_layers=$layers rlct_config.sampling_method=$sampler rlct_config.num_chains=$chains rlct_config.ed_config.eval_frequency=$ed_eval_f rlct_config.sgld_kwargs.lr=${rlct_lrs[$i]} rlct_config.sgld_kwargs.weight_decay=${weight_decays[$i]} rlct_config.use_distill_loss=$distill llc_train=$llc_train ed_train=$ed_train rlct_config.sgld_kwargs.mh_frequency=$mh_freq rlct_config.num_draws=$chain_len hydra.job.chdir=True
    done
done