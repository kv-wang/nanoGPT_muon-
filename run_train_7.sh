export WANDB_API_KEY="2eaf5d3e15da1d68fbce32137184e1eaba001ff6"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=2

torchrun --nproc_per_node=1 train_gpt_7.py --muon_type default --muon_momentum_0 0.95 --wandb no