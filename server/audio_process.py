from server.utils import get_mongo_client, get_file_path
import time
import json
import uuid
import subprocess
from loguru import logger

mongo_client = get_mongo_client()

db = mongo_client['ai_avatar']
ai_avatar_record_col = db['ai_avatar_record']
f = open('configs/system/digital_hunman_conf.json')
DHS = json.loads(f.read())
GPT_SOVotts_Dir = '/home/qc/workspace/GPT-SoVITS'
logger.info("音频处理开始")
while True:
    try:
        task = ai_avatar_record_col.find_one({'status': '0', 'content': {'$ne': None}})  # status = 1代表正在处理音频
        if not task:
            time.sleep(5)
            continue
        content = task.get('content')

        uid = uuid.uuid4().__str__()
        audio_output_path = get_file_path(uid + ".mp3")
        dh = DHS[task.get('video', {}).get('voice_name')]
        audio_conf = dh.get('audio_conf')
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
            cwd=GPT_SOVotts_Dir  # 指定工作目录
        )
        if status.returncode == 0:
            task['status'] = 1  # 音频处理完成
            task['voice_path'] = audio_output_path
            ai_avatar_record_col.update_one({'_id': task.get('_id')}, {'$set': task})

    except Exception as e:
        logger.error(e)
        break
    except KeyboardInterrupt:
        logger.info("音频处理关闭")
        break
    finally:
        pass
