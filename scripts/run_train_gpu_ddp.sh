#!/bin/bash
GPU_IDS="0,1,2,3"

OMP_NUM_THREADS=8 \
CUDA_VISIBLE_DEVICES=$GPU_IDS \
python3 -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 ./train.py \
    --output_dir="model_outputs/" \
    --data_dir="" \
    --seed=42 \
    --num_workers=12 \
    --per_device_train_batch_size=64 \
    --per_device_eval_batch_size=64 \
    --val_check_interval=0.25 \
    --accumulate_grad_batches=1 \
    --max_epochs=3 \
    --log_every_n_steps=1 \
    --accelerator=gpu \
    --strategy=ddp \
    --num_nodes=1 \
    --replace_sampler_ddp=false \
    --devices=4 \
    --auto_scale_batch_size=false \
    --learning_rate=0.00005 \
    --max_lr=0.0001 \
    --weight_decay=0.0001 \
    --warmup_ratio=0.2 \
    --ratio=0.2 \
    --div_factor=10 \
    --final_div_factor=10 \
    --valid_on_cpu=false \
    --model_select=rnn \
    --truncated_bptt_steps=1
