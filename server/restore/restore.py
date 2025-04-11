import time
import json
import torch
import tqdm
import numpy as np
import torchvision
from loguru import logger
from einops import rearrange
from latentsync.utils.image_processor import ImageProcessor
from latentsync.utils.util import write_video
from server.utils import affine_transform_video
import os
import uuid


class Restore:
    def __init__(self, redis_client, consume_queue, affine_col, ai_avatar_record_col):
        self.redis_client = redis_client
        self.consume_queue = consume_queue
        self.affine_col = affine_col
        self.image_processor = ImageProcessor(256, mask="fix_mask", device="cuda")
        self.ai_avatar_record_col = ai_avatar_record_col

    def restore_video(self, faces, video_frames, boxes, affine_matrices, start, num_frames: int = 16):
        out_frames = []
        print(f"Restoring {len(faces)} faces...")
        for index, face in enumerate(tqdm.tqdm(faces)):
            x1, y1, x2, y2 = boxes[(index + start * num_frames) % len(boxes)]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(face, size=(height, width), antialias=True)
            face = rearrange(face, "c h w -> h w c")
            face = (face / 2 + 0.5).clamp(0, 1)
            face = (face * 255).to(torch.uint8).cpu().numpy()
            # face = cv2.resize(face, (width, height), interpolation=cv2.INTER_LANCZOS4)
            out_frame = self.image_processor.restorer.restore_img(
                video_frames[(index + start * num_frames) % len(video_frames)],
                face,
                affine_matrices[
                    (index + start * num_frames) % len(affine_matrices)])
            out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)

    def run(self):
        while True:
            item = self.redis_client.lpop(self.consume_queue)
            if item is None:
                time.sleep(1)
                continue
            m = json.loads(item)
            video_path = m['video_path']
            start = m['start']
            pt_path = m['pt_path']
            sync_frames = torch.load(pt_path)

            affine_info = self.affine_col.find_one({'_id': video_path})
            faces, original_video_frames, boxes, affine_matrices = affine_transform_video(video_path, affine_info)
            synced_video_frames = self.restore_video(sync_frames, original_video_frames, boxes, affine_matrices, start)

            task_id = m['task_id']
            temp_dir = "temp_inf"
            video_out_path = os.path.join(temp_dir, f"{task_id}_{start}.npy")
            np.save(video_out_path, synced_video_frames)
            self.ai_avatar_record_col.update_one({'_id': task_id}, {
                '$set': {f'video_out_put.{start}': video_out_path, 'num_parts': m['num_parts']}})
            logger.info(m)
