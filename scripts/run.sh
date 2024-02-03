#!/bin/bash

# This script can optionally manage the main config name via command line passing to Hydra
SCRIPT_PATH="run.py"

lr="0.001,0.003,0.001,0.03"

python $SCRIPT_PATH -m optimizer_config.default_lr=$lr hydra.job.chdir=True

## To put straight into vscode:
# python scripts/run.py -m optimizer_config.default_lr=0.001,0.003,0.001,0.03 hydra.job.chdir=True
# python scripts/run.py -m optimizer_config.default_lr=0.001 hydra.job.chdir=True

# # ===================== TEACHER ONLY: RLCT ESTIMATION =========================
# SCRIPT_PATH="run_with_rlct.py"
# experiment="parity_transformer_teacher"

# n_hard=3
# p_simple="0.0,0.001,0.003,0.007,0.01,0.1,0.25,0.75,0.9,0.99,1"
# args="transformer_classifier_config.block_size=$((n_hard + 10))"

# for i in {1..1}
# do
#     HYDRA_FULL_ERROR=1 python $SCRIPT_PATH -m +experiment=$experiment dataset.parity_task_config.num_hard_tasks=$n_hard dataset.parity_task_config.prob_simple_task_on_train=$p_simple $args rlct_config.use_distill_loss=True rlct_config.save_results=True
# done


# # ===================== DISTILLATION: RLCT ESTIMATION =========================# 
# SCRIPT_PATH="run_distill_with_rlct.py"
# experiment="parity_transformer_distill"

# t_p_simple="0.0"
# n_hard=5
# p_simple="0.0,0.001,0.003,0.007,0.01"
# # p_simple="0.1,0.25,0.75,0.9,0.99,1"
# args="transformer_classifier_config.block_size=$((n_hard + 11))" # block_size should be the sum of num_hard_tasks (specifies num. control bits), num_hard_task_bits and num_simple_task_bits

# for i in {1..1}
# do
#     HYDRA_FULL_ERROR=1 python $SCRIPT_PATH -m +experiment=$experiment dataset.parity_task_config.num_hard_tasks=$n_hard dataset.parity_task_config.prob_simple_task_on_train=$p_simple teacher_name="${t_p_simple}_ht${n_hard}" teacher_model_artifact="iib-mech-robust/mechanistic-distillation/model:sp${t_p_simple}_ht${n_hard}" $args rlct_config.use_distill_loss=False
# done