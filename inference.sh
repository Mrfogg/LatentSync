 #!/bin/bash

python -m scripts.inference \
    --unet_config_path "configs/unet/second_stage.yaml" \
    --inference_ckpt_path "/home/qc/ai_server/ai_server/LatentSync/debug/unet/train-2025_03_02-16:02:19/checkpoints/checkpoint-1000.pt" \
    --inference_steps 20 \
    --guidance_scale 1.5 \
    --video_path "assets/demo1_video.mp4" \
    --audio_path "assets/demo1_audio.wav" \
    --video_out_path "video_out.mp4"
