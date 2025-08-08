#!/bin/bash

# Script to run sparse training on multiple GPUs (1, 2, 3)

echo "Starting multi-GPU training on GPUs 1, 2, 3..."

# Option 1: DataParallel (single process, multiple GPUs)
echo "Running with DataParallel..."
CUDA_VISIBLE_DEVICES=1,2,3 python sparse_train.py --dataset Chat --output_dir out_sparse --batch_size 6 --max_iters 5000

# Option 2: DistributedDataParallel (multiple processes)
# Uncomment the lines below to use DDP instead
# echo "Running with DistributedDataParallel..."
# CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 sparse_train.py --dataset Chat --output_dir out_sparse_ddp --batch_size 4 --max_iters 5000
