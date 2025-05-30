export WANDB_API_KEY=""
export TORCH_DISTRIBUTED_DEBUG=DETAIL
torchrun --standalone --nproc_per_node=4 train_gpt.py --muon_type default --warm_start 0.95 --wandb yes

start=0.5
end=0.9
step=0.1
for m in $(seq $start $step $end); do
  echo "Running with muon_momentum_0=$m"
  torchrun --standalone --nproc_per_node=4 \
    train_gpt.py \
    --muon_type default \
    --warm_start $m \
    --wandb yes
done

