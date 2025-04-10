import time

from server.utils import get_mongo_client, get_file_path
from latentsync.pipelines_multi.lipsync_pipeline_multi import affine_transform_video
from latentsync.pipelines_multi.utils import init_audio_encoder
from latentsync.utils.image_processor import ImageProcessor
from loguru import logger

mongo_client = get_mongo_client()

db = mongo_client['ai_avatar']
ai_avatar_record_col = db['ai_avatar_record']

audio_encoder = init_audio_encoder()
image_processor = ImageProcessor(256, mask="fix_mask", device="cuda")
gpu_num = 4
num_frames = 16


def whisper_feature(audio_encoder, audio_path):
    whisper_feature = audio_encoder.audio2feat(audio_path)
    whisper_chunks = audio_encoder.feature2chunks(feature_array=whisper_feature, fps=25)

    return whisper_chunks


while True:
    task = ai_avatar_record_col.find_one({'status': 1})
    video_path = task.get('video_path')
    affine_transform_video(video_path, image_processor)
    whisper_chunks = whisper_feature(audio_encoder, task.get('voice_path'))
    num_inferences = len(whisper_chunks) // num_frames
    if num_inferences > 10:
        num_parts = gpu_num
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
        logger.info(f"send data to subprocess {i}")
        sub_tasks[str(i)] = {'start': start, 'end': end}
        start = end
    task['sub_task'] = sub_tasks
    task['num_inferences'] = num_inferences
    task['status'] = 2  # 拆分任务结束
    ai_avatar_record_col.update_one({'_id': task.get('_id')}, {'$set': task})
    logger.info(f"sub tasks success video_path:{video_path}, task_id:{task.get('_id')}")
    time.sleep(10)
