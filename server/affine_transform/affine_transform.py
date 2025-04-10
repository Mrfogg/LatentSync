import json
import time

import os
import torch
import tqdm
import numpy as np
from latentsync.utils.image_processor import ImageProcessor
from latentsync.whisper.audio2feature import Audio2Feature
from latentsync.utils.util import read_video, read_audio, write_video
from server.const import GPU_NUM, SUB_INFERENCE_TASK
from loguru import logger
import  shutil

# affine_transform  拆分任务
def init_audio_encoder():
    whisper_model_path = "checkpoints/whisper/tiny.pt"
    audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=16)
    return audio_encoder


class AffineTransform:
    def __init__(self, redis_client, queue_name, affine_col):
        self.redis_client = redis_client
        self.queue_name = queue_name
        self.image_processor = ImageProcessor(256, mask="fix_mask", device="cuda")
        self.audio_encoder = init_audio_encoder()
        self.affine_cache_col = affine_col

    def whisper_feature(self, audio_encoder, audio_path):
        whisper_feature = audio_encoder.audio2feat(audio_path)
        whisper_chunks = audio_encoder.feature2chunks(feature_array=whisper_feature, fps=25)

        return whisper_chunks

    def affine_transform_video(self, video_path, image_processor):
        cache_path = '/data/data8T/video_process/' + os.path.basename(video_path) + '/'
        if self.affine_cache_col.find_one({'video_path': video_path}):
            return
        boxes_file_name = "box.dat"
        affine_matrix_file_name = "affine_matrix.dat"
        video_frames_file_name = "video_frames.dat"
        faces_file_name = "faces.pth"
        video_frames = read_video(video_path, use_decord=False)
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

        logger.info("开始写数据到磁盘")
        boxes = np.concatenate((boxes, boxes[::(-1)]))
        boxes_memmap = np.memmap(os.path.join(cache_path, boxes_file_name), mode='w+', shape=boxes.shape)
        boxes_memmap[:] = boxes
        boxes_memmap.flush()

        affine_matrices = np.concatenate((affine_matrices, affine_matrices[::-1]))
        affine_matrix_memmap = np.memmap(os.path.join(cache_path, affine_matrix_file_name), mode='w+',
                                         shape=affine_matrices.shape)
        affine_matrix_memmap[:] = affine_matrices
        affine_matrix_memmap.flush()

        faces = torch.stack(faces)
        f = torch.flip(faces, dims=(0,))
        faces = torch.cat((faces, f), dim=0)
        torch.save(faces, os.path.join(cache_path, faces_file_name))

        video_frames = video_frames[:len(faces) // 2]
        video_frames = np.concatenate((video_frames, video_frames[::-1]))
        video_frames_memmap = np.memmap(os.path.join(cache_path, video_frames_file_name), mode='w+',
                                        shape=video_frames.shape)
        video_frames_memmap[:] = video_frames
        video_frames_memmap.flush()
        logger.info("结束写数据到磁盘")
        self.affine_cache_col.insert_one({
            'video_path': video_path,
            'boxes_shape': boxes.shape,
            'affine_shape': affine_matrices.shape,
            'video_frame_shape': video_frames.shape,
            '_id': video_path,
        })
        return

    def run(self):
        while True:
            item = self.redis_client.lpop(self.queue_name)
            if item is None:
                time.sleep(1)
                continue
            m = json.loads(item)
            video_path = m['video_path']
            voice_path = m['voice_path']
            self.affine_transform_video(video_path, self.image_processor)

            whisper_chunks = self.whisper_feature(self.audio_encoder, voice_path)
            num_inferences = len(whisper_chunks) // 16
            if num_inferences > 10:
                num_parts = GPU_NUM
            else:
                num_parts = 1
            part_size = num_inferences // num_parts  # 每份的基础大小
            remainder = num_inferences % num_parts  # 剩余部分
            # 创建并启动多个进程
            start = 0
            logger.info(f"start distributed task:{num_parts}")
            sub_tasks = {}
            for i in range(num_parts):
                end = start + part_size + (1 if i < remainder else 0)
                sub_tasks[str(i)] = {'start': start, 'end': end}
                m['start'] = start
                m['end'] = end
                start = end
                self.redis_client.rpush(SUB_INFERENCE_TASK, json.dumps(m))
                logger.info(f"sub tasks success :{m}")
