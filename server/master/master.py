from loguru import logger
import requests
import time
from server.utils import download_file, upload_file_to_server
from server.const import *
import json
import subprocess
from datetime import datetime, timedelta
import os
import uuid
import numpy as np
from server.const import GPU_NUM
from latentsync.utils.util import write_video


class Master:
    def __init__(self, redis_client, avatar_record_col, next_queue_name):
        self.redis_client = redis_client
        self.avatar_record_col = avatar_record_col
        f = open('configs/system/digital_hunman_conf.json')
        DHS = json.loads(f.read())
        self.task_url = DHS.get('task_url')
        self.token = DHS.get('TOKEN')
        self.DHS = DHS
        self.notify_url = DHS.get('notify_url')
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
                if task.get('mode') != '普通模式':
                    continue
                mongo_task = self.avatar_record_col.find_one({'task_id': task.get('task_id')})
                if not mongo_task:
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
                    sub_result = mongo_task.get('video_out_put')
                    sub_videos = []
                    if sub_result and len(sub_result) == mongo_task.get('num_parts'):
                        logger.info(f"开始合并视频 {mongo_task.get('task_id')}")
                        for k in sorted(sub_result.keys(), key=lambda x: int(x)):
                            sub_videos.append(np.load(sub_result[k]))
                        out_video_frames = np.concatenate(sub_videos)
                        temp_dir = "temp_inf"
                        n = uuid.uuid4().__str__()
                        logger.info(f"结束numpy拼接视频 {mongo_task.get('task_id')}.mp4")
                        gen_video_out_path = os.path.join(temp_dir, n + ".mp4")
                        write_video(gen_video_out_path, out_video_frames, fps=25)

                        uid = uuid.uuid4().__str__()
                        video_output_path = f'/data/tmp/{uid}.mp4'
                        command = f"/home/qc/miniconda3/envs/latentsync/bin/ffmpeg -y -loglevel error -nostdin -i {gen_video_out_path} -i {mongo_task.get('voice_path')} -c:v libx264 -c:a aac -q:v 0 -q:a 0 {video_output_path}"
                        subprocess.run(command, shell=True)
                        logger.info(f"结束合并视频 {video_output_path}.mp4")
                        res = upload_file_to_server(self.notify_url, video_output_path,
                                                    {'taskid': task.get('task_id'), 'errcode': 0, 'status': 1})
                        logger.info(f"upload_file_to_server success: {res}")
