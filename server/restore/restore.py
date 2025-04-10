import time
import json
from loguru import logger


class Restore:
    def __init__(self, redis_client, consume_queue, affine_col):
        self.redis_client = redis_client
        self.consume_queue = consume_queue
        self.affine_col = affine_col

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
            logger.info(m)
