import time

import torch.multiprocessing as mp
from torch.multiprocessing import Queue
import os
import torch
import tqdm
import numpy as np
from .utils import init_pipeline
from ..utils.image_processor import ImageProcessor
from ..utils.util import read_video, read_audio, write_video
from .utils import init_audio_encoder
import soundfile as sf
from loguru import logger
import uuid
import subprocess

gpu_num = 4


def affine_transform_video(video_path, image_processor):
    cache_path = '/home/qc/data/video_preprocess/' + os.path.basename(video_path) + '/'
    faces_file_name = "faces.pth"
    boxes_file_name = "box.npy"
    affine_matrix_file_name = "affine_matrix.npy"
    video_frames = read_video(video_path, use_decord=False)

    if os.path.exists(cache_path):
        faces = torch.load(cache_path + faces_file_name)
        boxes = np.load(cache_path + boxes_file_name)
        affine_matrices = np.load(cache_path + affine_matrix_file_name)
        video_frames = video_frames[:len(faces)//2]
        video_frames = np.concatenate((video_frames, video_frames[::-1]))
        return faces, video_frames, boxes, affine_matrices
    os.makedirs(cache_path)
    faces = []
    boxes = []
    affine_matrices = []
    print(f"Affine transforming {len(video_frames)} faces...")
    i = 0
    for frame in tqdm.tqdm(video_frames):
        if i == len(video_frames) - len(video_frames) % 16:
            break
        i += 1
        face, box, affine_matrix = image_processor.affine_transform(frame)
        faces.append(face)
        boxes.append(box)
        affine_matrices.append(affine_matrix)

    video_frames = video_frames[:len(faces)]
    faces = torch.stack(faces)
    f = torch.flip(faces, dims=(0,))
    faces = torch.cat((faces, f), dim=0)
    torch.save(faces, cache_path + faces_file_name)

    boxes = np.concatenate((boxes, boxes[::(-1)]))
    video_frames = np.concatenate((video_frames, video_frames[::-1]))
    affine_matrices = np.concatenate((affine_matrices, affine_matrices[::-1]))
    np.save(cache_path + boxes_file_name, boxes)
    np.save(cache_path + affine_matrix_file_name, affine_matrices)
    return faces, video_frames, boxes, affine_matrices


def sub_process(in_queue, out_queue, config, args):
    processes = []
    for i in range(gpu_num):
        p = mp.Process(target=init_pipeline, args=(in_queue, out_queue, config, args, i))
        processes.append(p)
        p.start()
    return processes


class PipelineMaster:
    def __init__(self, config, args):
        mp.set_start_method('spawn', force=True)
        self.config = config
        self.args = args
        self.audio_encoder = init_audio_encoder()
        self.image_processor = ImageProcessor(config.data.resolution, mask="fix_mask", device="cuda")
        self.in_queue = Queue()
        self.out_queue = Queue()
        self.processes = sub_process(self.in_queue, self.out_queue, config, args)

    def whisper_feature(self, audio_path):
        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=25)

        return whisper_chunks

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def close(self):
        for i in range(gpu_num):
            self.in_queue.put(None)
    def process_video(self, video_path, audio_path, video_out_path, num_frames=16):
        logger.info(f"Processing video: {video_path} audio: {audio_path}")
        # audio_samples = read_audio(audio_path)
        affine_transform_video(video_path, self.image_processor)
        whisper_chunks = self.whisper_feature(audio_path)
        num_inferences = len(whisper_chunks) // num_frames
        if num_inferences > 10:
            num_parts = gpu_num
        else:
            num_parts = 1
        part_size = num_inferences // num_parts  # 每份的基础大小
        remainder = num_inferences % num_parts  # 剩余部分
        # 创建并启动多个进程
        start = 0
        logger.info(f"start distributed task:{num_parts}")
        for i in range(num_parts):
            end = start + part_size + (1 if i < remainder else 0)
            logger.info(f"send data to subprocess {i}")
            self.in_queue.put((video_path, audio_path, start, end))
            # 保存当前份的下标范围和子数组
            # 更新起始下标
            start = end
        result_sub_videos = []
        for i in range(num_parts):
            result = self.out_queue.get()
            sub_video = np.load(result[0])
            logger.info(f"save sub video {result[0]} shape:{sub_video.shape}")
            result_sub_videos.append((sub_video, result[1]))
        result_sub_videos = sorted(result_sub_videos, key=lambda x: x[1])
        out_video_frames = np.concatenate([sub_video[0] for sub_video in result_sub_videos])
        temp_dir = "temp_inf"
        n = uuid.uuid4().__str__()
        gen_video_out_path = os.path.join(temp_dir, n + ".mp4")
        write_video(gen_video_out_path, out_video_frames, fps=25)
        command = f"/home/qc/miniconda3/envs/latentsync/bin/ffmpeg -y -loglevel error -nostdin -i {gen_video_out_path} -i {audio_path} -c:v libx264 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)
        logger.info(f"Processing video success:output_video_path {video_out_path}")
