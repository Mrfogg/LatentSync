#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3 torchrun --nnodes=1 --nproc_per_node=2 --master_port=25678 -m scripts.train_unet \
    --unet_config_path "configs/unet/first_stage.yaml"
