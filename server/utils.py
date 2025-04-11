import os
from datetime import datetime
import requests
from urllib.parse import urlparse
from loguru import logger
from pymongo import MongoClient
import torch
import numpy as np


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


def get_file_path(file_name):
    today_date = datetime.now().strftime("%Y-%m-%d")
    folder_path = os.path.join('/data/data8T', today_date)
    return os.path.join(folder_path, file_name)


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


def get_mongo_client():
    mlient = MongoClient('mongodb://192.168.1.9:27017/')
    return mlient


def affine_transform_video(video_path, affine_info):
    logger.info(f"affine transforming video {video_path}...")
    cache_path = '/data/data8T/video_process/' + os.path.basename(video_path) + '/'
    boxes_file_name = "box.dat"
    affine_matrix_file_name = "affine_matrix.dat"
    video_frames_file_name = "video_frames.dat"
    faces_file_name = "faces.pth"
    logger.info(affine_info)
    if os.path.exists(cache_path):
        boxes_memmap = np.memmap(os.path.join(cache_path, boxes_file_name), mode='r',
                                 shape=tuple(affine_info['boxes_shape']),dtype=np.uint32)

        affine_matrix_memmap = np.memmap(os.path.join(cache_path, affine_matrix_file_name), mode='r',
                                         shape=tuple(affine_info['affine_shape']), dtype=np.float64)
        faces = torch.load(os.path.join(cache_path, faces_file_name))

        video_frames_memmap = np.memmap(os.path.join(cache_path, video_frames_file_name), mode='r',
                                        shape=tuple(affine_info['video_frame_shape']))
        return faces, video_frames_memmap, boxes_memmap, affine_matrix_memmap
    return None


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
