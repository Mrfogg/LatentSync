import argparse
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature


class DigitalHuman:
    def __init__(self, video_path, sync_model_path, unet_model_path):
        self.video_path = video_path
        self.sync_model_path = sync_model_path
        self.unet_model_path = unet_model_path


    def initialize(self, config, args):
        is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
        dtype = torch.float16 if is_fp16_supported else torch.float32

        print(f"Input video path: {args.video_path}")
        print(f"Input audio path: {args.audio_path}")
        print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

        self.scheduler = DDIMScheduler.from_pretrained("configs")

        if config.model.cross_attention_dim == 768:
            whisper_model_path = "checkpoints/whisper/small.pt"
        elif config.model.cross_attention_dim == 384:
            whisper_model_path = "checkpoints/whisper/tiny.pt"
        else:
            raise NotImplementedError("cross_attention_dim must be 768 or 384")

        self.audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0

        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            args.inference_ckpt_path,  # load checkpoint
            device="cpu",
        )

        unet = unet.to(dtype=dtype)

        # set xformers
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()

        pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=self.audio_encoder,
            unet=unet,
            scheduler=self.scheduler,
        ).to("cuda")
        if args.seed != -1:
            set_seed(args.seed)
        else:
            torch.seed()

        print(f"Initial seed: {torch.initial_seed()}")
        print(args.inference_steps, "steps")
        pipeline(
            video_path=args.video_path,
            audio_path=args.audio_path,
            video_out_path=args.video_out_path,
            video_mask_path=args.video_out_path.replace(".mp4", "_mask.mp4"),
            num_frames=config.data.num_frames,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            weight_dtype=dtype,
            width=config.data.resolution,
            height=config.data.resolution,
        )
