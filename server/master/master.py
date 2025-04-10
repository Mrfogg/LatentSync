from loguru import logger
import requests
import time
from server.utils import download_file
from server.const import *
import json
import subprocess
from datetime import datetime, timedelta
import os
import uuid


class Master:
    def __init__(self, redis_client, avatar_record_col, next_queue_name):
        self.redis_client = redis_client
        self.avatar_record_col = avatar_record_col
        f = open('configs/system/digital_hunman_conf.json')
        DHS = json.loads(f.read())
        self.task_url = DHS.get('task_url')
        self.token = DHS.get('TOKEN')
        self.DHS = DHS
        self.GPT_SOVotts_Dir = '/home/qc/workspace/GPT-SoVITS'
        self.next_queue_name = next_queue_name

    def get_file_path(self, file_name):
        return os.path.join(TMP_DIR, file_name)

    def audio_generate(self, task):
        dh = self.DHS[task.get('video', {}).get('voice_name')]
        audio_conf = dh.get('audio_conf')
        uid = uuid.uuid4().__str__()
        audio_output_path = self.get_file_path(uid + ".mp3")
        status = subprocess.run(
            ['/home/qc/miniconda3/envs/GPTSoVits/bin/python', "GPT_SoVITS/inference_cli.py", "--gpt_model",
             audio_conf.get('gpt_model'), '--sovits_model',
             audio_conf.get('sovits_model'),
             '--ref_audio',
             '/home/qc/ai_server/ai_server/assets/数字人/%s/%s' % (
                 dh.get('name'), audio_conf.get('ref_audio')),
             '--ref_text', audio_conf.get('ref_text'),
             '--ref_language', '中文', '--target_text', task.get('content'), '--target_language', '中文',
             '--output_path', audio_output_path, '--speed', "%.1f" % audio_conf.get('speed'), '--inp_refs',
             audio_conf.get('inp_refs')],  # 要运行的命令和参数
            text=True,  # 以文本形式处理输出
            cwd=self.GPT_SOVotts_Dir  # 指定工作目录
        )
        if status.returncode == 0:
            task['voice_path'] = audio_output_path
            return True
        return False

    def run(self):
        while True:
            time.sleep(1)
            task_data = requests.get(self.task_url, headers={'token': self.token}).json()
            if task_data['code'] == -1:
                logger.info(task_data['msg'])
                break
            if task_data.get("data", {}).get("count", 0) == 0:
                time.sleep(2)
                continue
            lists = task_data.get("data", {}).get("lists", [])
            for task in lists[::-1]:
                if task.get('mode') != '普通模式' or task.get('account') != '13002090253':
                    continue
                if not self.avatar_record_col.find_one({'task_id': task.get('task_id')}):
                    task['_id'] = task.get('task_id')
                    video_url = task.get('video', {}).get('video_url', '')
                    video_path = download_file(video_url, TEMPPLATE_VIDEO_PATH)
                    task['video_path'] = video_path
                    if task.get('voice_url', '') != '':
                        voice_path = download_file(task.get('voice_url', ''), TMP_DIR)
                        task['voice_path'] = voice_path
                    else:
                        if not self.audio_generate(task):
                            continue
                    task['createdAt'] = datetime.utcnow()
                    self.avatar_record_col.insert_one(task)
                    msg = {
                        'task_id': task.get('task_id'),
                        'video_path': task.get('video_path'),
                        'voice_path': task.get('voice_path'),
                    }
                    self.redis_client.rpush(
                        self.next_queue_name, json.dumps(msg))
                    logger.info(f"lpush queue {self.next_queue_name} task_id: {task.get('task_id')}")

                else:
                    pass  # 这里写检测逻辑
