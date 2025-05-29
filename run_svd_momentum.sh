export WANDB_API_KEY="2eaf5d3e15da1d68fbce32137184e1eaba001ff6"
export TORCH_DISTRIBUTED_DEBUG=DETAIL

torchrun --standalone --nproc_per_node=4 train_gpt.py --muon_type svd_momentum --muon_momentum_0 0.0 --wandb yes
torchrun --standalone --nproc_per_node=4 train_gpt.py --muon_type svd_momentum --muon_momentum_0 1.0 --wandb yes


start=0.75
end=0.95
step=0.05
for m in $(seq $start $step $end); do
  echo "Running with muon_momentum_0=$m"
  torchrun --standalone --nproc_per_node=4 \
    train_gpt.py \
    --muon_type svd_momentum \
    --muon_momentum_0 $m \
    --wandb yes
done

start=0.96
end=0.99
step=0.01
for m in $(seq $start $step $end); do
  echo "Running with muon_momentum_0=$m"
  torchrun --standalone --nproc_per_node=4 \
    train_gpt.py \
    --muon_type svd_momentum \
    --muon_momentum_0 $m \
    --wandb yes
done