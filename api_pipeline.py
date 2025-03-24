import json
from omegaconf import OmegaConf
import time
import uuid
import torch
from latentsync.pipelines_multi.lipsync_pipeline_multi import PipelineMaster
from dataclasses import dataclass
import requests
import subprocess
import json
from loguru import logger
from feiying.ai import generate_video_task, query_task
from urllib.parse import urlparse
TASK_URL = 'http://192.144.152.86/tenantapi/avatar.aiAvatarRecord/lists?page_no=1&page_size=10&user_info=&status=0'
TOKRN = '69a6575adfbc6ff9c2250928d8aa6634'
f = open('configs/system/digital_hunman_conf.json')
DHS = json.loads(f.read())
notify_url = 'http://192.144.152.86/api/avatar.aiAvatarRecord/receiveAiAvatar'


@dataclass
class Para:
    pass


import requests
import os


def get_filename_from_url(url):
    """
    从 URL 中提取文件名，并去掉查询字符串和锚点。

    参数:
        url (str): 输入的 URL。

    返回:
        str: 提取的文件名。
    """
    # 解析 URL
    parsed_url = urlparse(url)

    # 获取路径部分
    path = parsed_url.path

    # 使用 os.path.basename 获取文件名
    from os.path import basename
    filename = basename(path)

    return filename

def download_file(url, save_path=".", filename=None):
    """
    下载视频文件到本地。

    :param url: 视频的 URL 地址
    :param save_path: 保存视频的本地目录，默认为当前工作目录
    :param filename: 保存的文件名，如果不指定，则使用 URL 的最后一部分作为文件名
    """
    try:
        # 发送 HTTP GET 请求获取视频内容
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功

        # 如果没有指定文件名，使用 URL 的最后一部分
        if filename is None:
            filename = get_filename_from_url(url)

        # 构造完整的保存路径
        full_path = os.path.join(save_path, filename)

        # 以二进制写入模式打开文件
        with open(full_path, "wb") as file:
            # 分块下载，避免占用过多内存
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        logger.debug(f"视频已成功下载到 {full_path}")
        return full_path
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        return ''


GPT_SOVotts_dir = '/home/qc/workspace/GPT-SoVITS'


def upload_file_to_server(url, file_path, other_params):
    """
    上传文件到服务器的函数

    :param url: 服务器的上传接口 URL
    :param file_path: 要上传的文件路径
    :param other_params: 其他参数，字典形式
    :return: 服务器返回的响应内容
    """
    # 打开文件
    data = other_params
    if os.path.exists(file_path) and file_path:
        with open(file_path, 'rb') as file:
            # 构造文件上传的字典
            files = {
                'target_file': (file_path, file),
            }

            # 发送 POST 请求
            response = requests.post(url, files=files, data=data)
    else:
        response = requests.post(url, data=data)

    # 返回响应内容
    return response.json()  # 假设服务器返回的是 JSON 格式


sub_companys = ['17737711610', '18037328992', '拿摩万团队-良师', 'kanghualan888',
                '15838310311', '18639873957', '17737706163',
                '18838706389', '15629178657', '18595468330', '15638397074','13002090253']
audio_out_path_format = '/home/qc/ai_server/ai_server/upload/%s.wav'
audio_url_format = 'http://82.157.200.241:8080/humanmeta_file/%s.wav'

if __name__ == '__main__':
    args = Para()
    args.unet_config_path = "configs/unet/second_stage_prod.yaml"
    args.inference_ckpt_path = "/home/qc/data/model/checkpoint-31000.pt"
    args.seed = 1
    config = OmegaConf.load(args.unet_config_path)
    pm = PipelineMaster(config, args)  #
    logger.info("数字人系统启动")
    while True:
        try:
            torch.cuda.empty_cache()
            task_data = requests.get(TASK_URL, headers={'token': TOKRN}).json()
            if task_data['code'] == -1:
                logger.info(task_data['msg'])
                pm.close()
                break
            if task_data.get("data", {}).get("count", 0) == 0:
                time.sleep(2)
                continue
            lists = task_data.get("data", {}).get("lists", [])
            task = lists[::-1][0]
            if task.get('content') != "":  # 文本合成
                dh = DHS[task.get('video', {}).get('voice_name')]
                audio_conf = dh.get('audio_conf')
                uid = uuid.uuid4().__str__()
                audio_output_path = audio_out_path_format % uid
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
                    cwd=GPT_SOVotts_dir  # 指定工作目录
                )
                if status.returncode == 0:
                    if task.get('account') in sub_companys:
                        job_id = generate_video_task(task.get('video').get('video_url'), audio_url_format % uid)
                        while True:
                            time.sleep(10)  # 一直等待
                            video_url, status = query_task(job_id)
                            if status == 3:  # 作品完成
                                fy_video_path = download_file(video_url, '/data/tmp')
                                res = upload_file_to_server(notify_url, fy_video_path,
                                                            {'taskid': task.get('task_id'), 'errcode': 0,
                                                             'status': 1})
                                break
                            elif status == 4:
                                res = upload_file_to_server(notify_url, "",
                                                            {'taskid': task.get('task_id'), 'errcode': 500,
                                                             'status': 1})
                                break
                    else:
                        full_path = download_file(task.get('video').get('video_url'), f'/data/tmp')
                        video_output_path = f'/data/tmp/{uid}.mp4'
                        pm.process_video(full_path, audio_output_path, video_output_path)
                        res = upload_file_to_server(notify_url, video_output_path,
                                                    {'taskid': task.get('task_id'), 'errcode': 0, 'status': 1})
                else:
                    pass
            if task.get('voice_url', '') != '':
                if task.get('account') in sub_companys:
                    job_id = generate_video_task(task.get('video').get('video_url'), task.get('voice_url'))
                    while True:
                        time.sleep(10)  # 一直等待
                        video_url, status = query_task(job_id)
                        if status == 3:  # 作品完成
                            fy_video_path = download_file(video_url, '/data/tmp')
                            res = upload_file_to_server(notify_url, fy_video_path,
                                                        {'taskid': task.get('task_id'), 'errcode': 0, 'status': 1})
                            break
                        elif status == 4:
                            res = upload_file_to_server(notify_url, "",
                                                        {'taskid': task.get('task_id'), 'errcode': 500,
                                                         'status': 1})
                            break
                else:
                    voice_path = download_file(task.get('voice_url'), f'/data/tmp')
                    full_path = download_file(task.get('video').get('video_url'), f'/data/tmp')
                    uid = uuid.uuid4().__str__()
                    video_output_path = f'/data/tmp/{uid}.mp4'
                    pm.process_video(full_path, voice_path, video_output_path)
                    res = upload_file_to_server(notify_url, video_output_path,
                                                {'taskid': task.get('task_id'), 'errcode': 0, 'status': 1})

        except KeyboardInterrupt:
            pm.close()
            logger.info("数字人系统关闭")

        except Exception as e:
            res = upload_file_to_server(notify_url, "",
                                        {'taskid': task.get('task_id'), 'errcode': 500, 'status': 1})
            logger.error(e)
