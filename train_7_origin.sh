export CUDA_VISIBLE_DEVICES=2,3,4,5
torchrun --nproc_per_node=4 train_gpt_7_origin.py