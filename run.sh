#!/bin/bash

export WANDB_API_KEY="2eaf5d3e15da1d68fbce32137184e1eaba001ff6"
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 声明项目名称变量
PROJECT_NAME="run_log_23_sku3"

# 运行不同的实验
torchrun --standalone --nproc_per_node=8 train_gpt_23.py --muon_type default --muon_momentum_0 0.95 --wandb yes --tensorboard_path $PROJECT_NAME --wandb_project $PROJECT_NAME

torchrun --standalone --nproc_per_node=8 train_gpt_23.py --muon_type double_momentum --muon_momentum_0 0.95 --muon_momentum_1 0.98 --wandb yes --tensorboard_path $PROJECT_NAME --wandb_project $PROJECT_NAME

torchrun --standalone --nproc_per_node=8 train_gpt_23.py --muon_type default --adaptive_beta yes --muon_momentum_0 0.95 --wandb yes --tensorboard_path $PROJECT_NAME --wandb_project $PROJECT_NAME

torchrun --standalone --nproc_per_node=8 train_gpt_23.py --muon_type adam --wandb yes --tensorboard_path $PROJECT_NAME --wandb_project $PROJECT_NAME

torchrun --standalone --nproc_per_node=8 train_gpt_23.py --muon_type double_momentum --muon_momentum_0 0.90 --muon_momentum_1 0.95 --wandb yes --tensorboard_path $PROJECT_NAME --wandb_project $PROJECT_NAME

torchrun --standalone --nproc_per_node=8 train_gpt_23.py --muon_type svd_momentum_v2 --muon_momentum_0 0.95 --wandb yes --tensorboard_path $PROJECT_NAME --wandb_project $PROJECT_NAME 