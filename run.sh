export WANDB_API_KEY="2eaf5d3e15da1d68fbce32137184e1eaba001ff6"
export TORCH_DISTRIBUTED_DEBUG=DETAIL

torchrun --standalone --nproc_per_node=4 train_gpt_grad_accu.py --muon_type default --wandb no
#torchrun --standalone --nproc_per_node=4 train_gpt.py --muon_type double_momentum
#torchrun --standalone --nproc_per_node=4 train_gpt.py --muon_type default
#torchrun --standalone --nproc_per_node=4 train_gpt.py --muon_type svd_momentum


