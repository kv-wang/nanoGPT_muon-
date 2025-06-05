export WANDB_API_KEY=""
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=2,3,4,5

torchrun --standalone --nproc_per_node=4 \
  train_gpt.py \
  --muon_type default \
  --muon_momentum_0 0.95 \
  --muon_momentum_1 0.95 \
  --wandb no



start0=0.80
end0=0.92
step0=0.02

start1=0.93
end1=0.99
step1=0.02

# 使用 awk 来生成浮点数的 range
for m0 in $(seq $start0 $step0 $end0); do
  for m1 in $(seq $start1 $step1 $end1); do
    echo "Running with muon_momentum_0=$m0, muon_momentum_1=$m1"
    torchrun --standalone --nproc_per_node=4 \
      train_gpt.py \
      --muon_type double_momentum \
      --muon_momentum_0 $m0 \
      --muon_momentum_1 $m1 \
      --wandb no
  done
done


start=0.85
end=0.99
step=0.02
for m in $(seq $start $step $end); do
  echo "Running with muon_momentum_0=$m"
  torchrun --standalone --nproc_per_node=4 \
    train_gpt.py \
    --muon_type svd_momentum_v2 \
    --muon_momentum_0 $m \
    --wandb no
done