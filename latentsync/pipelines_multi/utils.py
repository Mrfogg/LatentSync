import torch
from diffusers import DDIMScheduler, AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf

from latentsync.models.unet import UNet3DConditionModel
from latentsync.utils.image_processor import ImageProcessor
from .subprocess import LipsyncPipeline
from accelerate.utils import set_seed

from latentsync.whisper.audio2feature import Audio2Feature

is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
dtype = torch.float16 if is_fp16_supported else torch.float32
num_frames = 16


def init_image_processor(height, mask):
    image_processor = ImageProcessor(height, mask=mask, device="cuda")
    return image_processor


def init_audio_encoder():
    whisper_model_path = "checkpoints/whisper/tiny.pt"
    audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=num_frames)
    return audio_encoder


def init_pipeline(in_queue, out_queue, config, args,rank) -> LipsyncPipeline:
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

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
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
        in_queue=in_queue,
        out_queue=out_queue,
        rank=rank,
    ).to("cuda:"+str(rank%3))
    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()
    pipeline.run(rank)
