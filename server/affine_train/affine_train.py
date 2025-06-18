from loguru import logger
import requests
from server.const import *
import json
import time

import os

from server.utils import download_file, get_filename_from_url
from server.const import IMAGE_URL, TEMPPLATE_VIDEO_PATH, RECIEVE_IMAGE_URL, TOKEN


class AffineTrainMaster:
    def __init__(self, redis_client, next_queue_name, affine_col, affine_train_col):
        self.redis_client = redis_client
        self.next_queue_name = next_queue_name
        self.affine_col = affine_col
        self.affine_train_col = affine_train_col
        self.cache = []

    def get_file_path(self, file_name):
        return os.path.join(TMP_DIR, file_name)

    def run(self):
        while True:
            time.sleep(1)
            task_data = requests.get(IMAGE_URL, headers={'token': TOKEN}).json()
            if task_data['code'] == -1:
                logger.info(task_data['msg'])
                break
            if task_data.get("data", {}).get("count", 0) == 0:
                time.sleep(2)
                continue
            lists = task_data.get("data", {}).get("lists", [])
            for image in lists[::-1]:
                mongo_image = self.affine_train_col.find_one({'_id': image.get('sn')})
                if not mongo_image:
                    video_path = download_file(image.get('fileUrl'), TEMPPLATE_VIDEO_PATH)
                    self.affine_col.delete_one({'_id': video_path})
                    if video_path == '':
                        logger.error(f"下载失败url:{image.get('fileUrl')}")
                        continue
                    msg = {
                        'id': image.get('id'),
                        'video_path': video_path,
                        'type': 'affine_train',
                    }
                    self.redis_client.rpush(
                        self.next_queue_name, json.dumps(msg))
                    image['video_path'] = video_path
                    image['_id'] = image.get('sn')
                    self.affine_train_col.insert_one(image)
                    logger.info(f"lpush queue {self.next_queue_name} video_path: {msg}")
                else:
                    transform_cache = self.affine_col.find_one({'_id': mongo_image.get('video_path')})
                    if not transform_cache:
                        continue
                    if transform_cache.get('status') == 1:
                        res = requests.post(RECIEVE_IMAGE_URL,
                                            data={'id': image.get('id'), 'errcode': 0, 'status': 1,
                                                  'mode_id': 24})  # model_id没用，只是符合服务校验格式
                        logger.info(f"affine_transform success: {mongo_image}, {res.json()}")
                    elif transform_cache.get('status') == 2:
                        res = requests.post(RECIEVE_IMAGE_URL,
                                            data={'id': image.get('id'), 'errcode': 400, 'status': 2, 'mode_id': 24})
                        logger.error(f"affine_transform err: {mongo_image}, {res.json()}")
