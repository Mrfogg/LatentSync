#!/bin/bash

torchrun --nnodes=1 --nproc_per_node=4 --master_port=25679 -m scripts.train_syncnet \
    --config_path "configs/syncnet/syncnet_16_pixel.yaml"
