import json
import time
import uuid

from dataclasses import dataclass
import requests
import subprocess
import json
from loguru import logger
from urllib.parse import urlparse

TASK_URL = 'https://suibai.vip/tenantapi/voicerecord.voiceRecord/lists?/lists?page_no=1&page_size=10&user_info=&status=0'
TOKRN = '9ed86c716c2e948098b247b01a93190f'
f = open('configs/system/digital_hunman_conf.json')
DHS = json.loads(f.read())
notify_url = 'https://suibai.vip/api/voice.record/receiveCloneVoice'


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


audio_out_path_format = '/home/qc/ai_server/ai_server/upload/%s.wav'

if __name__ == '__main__':
    logger.info("音频克隆系统启动")
    while True:
        try:
            task_data = requests.get(TASK_URL, headers={'token': TOKRN}).json()
            if task_data['code'] == -1:
                logger.info(task_data['msg'])
                break
            if task_data.get("data", {}).get("count", 0) == 0:
                time.sleep(2)
                continue
            lists = task_data.get("data", {}).get("lists", [])
            for task in lists[::-1]:
                if task.get('content') != "":  # 文本合成
                    dh = DHS[task.get('timbre_name', '')]
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
                        res = upload_file_to_server(notify_url, audio_output_path,
                                                    {'taskid': task.get('task_id'), 'errcode': 0, 'status': 1})
                        print(res)
                    else:
                        pass

        except KeyboardInterrupt:
            logger.info("音频系统关闭")
            break

        except Exception as e:
            logger.error(e)
            break
