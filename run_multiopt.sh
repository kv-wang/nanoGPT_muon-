export WANDB_API_KEY="2eaf5d3e15da1d68fbce32137184e1eaba001ff6"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=2,3

torchrun --standalone --nproc_per_node=2 train_gpt_multiopt.py \
    --muon_type_1 default \
    --cfg_1 0.95 0.95 \
    --muon_type_2 default \
    --cfg_2 0.95 0.95 \
    --split 0.4

torchrun --standalone --nproc_per_node=2 train_gpt_multiopt.py \
    --muon_type_1 double_momentum \
    --cfg_1 0.90 0.95 \
    --muon_type_2 double_momentum \
    --cfg_2 0.90 0.99 \
    --split 0.4

torchrun --standalone --nproc_per_node=2 train_gpt_multiopt.py \
    --muon_type_1 double_momentum \
    --cfg_1 0.90 0.99 \
    --muon_type_2 double_momentum \
    --cfg_2 0.90 0.95 \
    --split 0.4

torchrun --standalone --nproc_per_node=2 train_gpt_multiopt.py \
    --muon_type_1 double_momentum \
    --cfg_1 0.90 0.95 \
    --muon_type_2 double_momentum \
    --cfg_2 0.90 0.95 \
    --split 0.4

torchrun --standalone --nproc_per_node=2 train_gpt_multiopt.py \
    --muon_type_1 double_momentum \
    --cfg_1 0.80 0.95 \
    --muon_type_2 double_momentum \
    --cfg_2 0.90 0.95 \
    --split 0.4

torchrun --standalone --nproc_per_node=4 train_gpt_multiopt.py \
    --muon_type_1 double_momentum \
    --cfg_1 0.90 0.95 \
    --muon_type_2 double_momentum \
    --cfg_2 0.80 0.95 \
    --split 0.4

torchrun --standalone --nproc_per_node=2 train_gpt_multiopt.py \
    --muon_type_1 default \
    --cfg_1 0.95 0.95 \
    --muon_type_2 double_momentum \
    --cfg_2 0.90 0.95 \
    --split 0.4

torchrun --standalone --nproc_per_node=2 train_gpt_multiopt.py \
    --muon_type_1 default \
    --cfg_1 0.95 0.95 \
    --muon_type_2 svd_momentum_v2 \
    --cfg_2 0.95 0.95 \
    --split 0.4