import torch
from diffusers import DDIMScheduler, AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
import time
from latentsync.models.unet import UNet3DConditionModel
from latentsync.utils.image_processor import ImageProcessor
from server.inference.subprocess import LipsyncPipelineSubprocess
from accelerate.utils import set_seed
import json
from latentsync.whisper.audio2feature import Audio2Feature

is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
dtype = torch.float16 if is_fp16_supported else torch.float32
num_frames = 16
from loguru import logger


class Inference:
    def __init__(self, redis_client, consume_queue, gpu_id, restore_queue_name, affine_col):
        self.redis_client = redis_client
        self.consume_queue = consume_queue
        self.gpu_id = gpu_id
        self.restore_queue_name = restore_queue_name
        self.process = self.init_pipeline('checkpoints/latentsync_unet.pt')
        self.affine_col = affine_col

    def init_pipeline(self, inference_ckpt_path) -> LipsyncPipelineSubprocess:
        config = OmegaConf.load("configs/unet/second_stage_prod.yaml")
        is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
        dtype = torch.float16 if is_fp16_supported else torch.float32

        scheduler = DDIMScheduler.from_pretrained("configs")

        whisper_model_path = "checkpoints/whisper/tiny.pt"

        audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0

        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            inference_ckpt_path,  # load checkpoint
            device="cpu",
        )

        unet = unet.to(dtype=dtype)

        # set xformers
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        pipeline = LipsyncPipelineSubprocess(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        ).to("cuda:" + str(self.gpu_id % 4))
        torch.seed()
        return pipeline

    def run(self):
        while True:
            item = self.redis_client.lpop(self.consume_queue)
            if item is None:
                time.sleep(1)
                continue
            m = json.loads(item)
            video_path = m['video_path']
            voice_path = m['voice_path']
            start = m['start']
            end = m['end']
            affine_info = self.affine_col.find_one({'_id': video_path})
            logger.info(f"start inference: {video_path}, {voice_path}, {start}, {end}")
            pt_path = self.process.inference_batch(video_path, voice_path, start, end,affine_info)
            m['pt_path'] = pt_path
            self.redis_client.rpush(self.restore_queue_name, json.dumps(m))
            logger.info(m)
