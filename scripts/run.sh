#!/bin/bash

# This script can optionally manage the main config name via command line passing to Hydra
SCRIPT_PATH="run.py"

task="dihedral"

# Whether to log particular things
llc_train="False"
ed_train="True"

# Hyperparams
lr=1e-3
its=10000
eval_f=100
layers=3
seq_len=25

# RLCT
chains=5
ed_eval_f=3
rlct_lr=1e-8
weight_decay=1e-8
chain_len=3500

for i in {1..1}
do
    HYDRA_FULL_ERROR=1 python $SCRIPT_PATH -m num_training_iter=$its optimizer_config.default_lr=$lr eval_frequency=$eval_f task_config=$task ++task_config.length=$seq_len nano_gpt_config.n_layers=$layers rlct_config.num_chains=$chains rlct_config.ed_config.eval_frequency=$ed_eval_f rlct_config.sgld_kwargs.lr=$rlct_lr rlct_config.sgld_kwargs.weight_decay=$weight_decay llc_train=$llc_train ed_train=$ed_train rlct_config.num_draws=$chain_len hydra.job.chdir=True

done