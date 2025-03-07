import sys
from scripts.inference import main
from dataclasses import dataclass
from peewee import MySQLDatabase, Model, CharField, IntegerField
from server.queue import db
import os
import uuid
from omegaconf import OmegaConf
@dataclass
class Para:
    pass


import requests


def download_file(url, save_path):
    try:
        # 发送 HTTP GET 请求
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功

        # 将文件写入本地
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"文件已下载到: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")


base_out_path = '/home/qc/ai_server/ai_server/upload/'


class FileVideo(Model):
    id = IntegerField(primary_key=True)
    uri = CharField()

    class Meta:
        database = db
        table_name = '192_tenant_file'  # 表名


def get_video_url_and_download(file_id):
    file = FileVideo.select().where(FileVideo.id == file_id).first()
    path = base_out_path + os.path.basename(file.uri)
    if not os.path.exists(path):
        file_url = 'http://182.123.101.199/%s' % file.save_path
        download_file(file_url, path)
    return path


def generate_video_u(file_id, audio_url, queue):
    args = Para()
    id = uuid.uuid4().__str__()
    args.unet_config_path = "configs/unet/second_stage.yaml"
    args.inference_ckpt_path = "debug/unet/train-2025_03_02-19:22:37/checkpoints/checkpoint-20000.pt"
    args.seed = 1
    config = OmegaConf.load(args.unet_config_path)
    audio_path = base_out_path + os.path.basename(audio_url)
    print(file_id)
    args.video_path = get_video_url_and_download(file_id)
    args.audio_path = audio_path
    args.video_out_path = base_out_path + id + '.mp4'
    args.guidance_scale = 2.5
    args.inference_steps = 40
    args.gpu_id = 2
    main(config, args, queue)
    return args.video_out_path
