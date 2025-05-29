export WANDB_API_KEY=""
export TORCH_DISTRIBUTED_DEBUG=DETAIL

torchrun --standalone --nproc_per_node=1 train_gpt_multiopt.py \
    --muon_type_1 default \
    --cfg_1 0.95 0.95 \
    --muon_type_2 default \
    --cfg_2 0.95 0.95 \
    --split 0.4