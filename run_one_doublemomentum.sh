export WANDB_API_KEY="2eaf5d3e15da1d68fbce32137184e1eaba001ff6"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=2,3,4,5


torchrun --nproc_per_node=4 train_gpt_7.py --muon_type default --muon_momentum_0 0.95 --wandb no --tensorboard_path one_doublemomentum --wandb_project one_doublemomentum
torchrun --nproc_per_node=4 train_gpt_7.py --muon_type adam --wandb no --tensorboard_path one_doublemomentum --wandb_project one_doublemomentum
torchrun --nproc_per_node=4 train_gpt_7.py --muon_type double_momentum --muon_momentum_0 0.95 --muon_momentum_1 0.98 --wandb no --tensorboard_path one_doublemomentum --wandb_project one_doublemomentum
torchrun --nproc_per_node=4 train_gpt_7.py --muon_type double_momentum --muon_momentum_0 0.95 --muon_momentum_1 0.99 --wandb no --tensorboard_path one_doublemomentum --wandb_project one_doublemomentum
torchrun --nproc_per_node=4 train_gpt_7.py --muon_type svd_momentum_v2 --muon_momentum_0 0.95 --wandb no --tensorboard_path one_doublemomentum --wandb_project one_doublemomentum


