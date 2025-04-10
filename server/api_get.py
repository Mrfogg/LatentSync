import requests
import time
from loguru import logger
from server.utils import download_file, get_mongo_client
from server.const import *
import redis

TASK_URL = 'https://suibai.vip/tenantapi/avatar.aiAvatarRecord/lists?page_no=1&page_size=10&user_info=&status=0'
TOKRN = '9ed86c716c2e948098b247b01a93190f'
mlient = get_mongo_client()
db = mlient['ai_avatar']
ai_avatar_record_col = db['ai_avatar_record']
account_col = db['feiying_user']

r = redis.Redis(host='localhost', port=6379, db=0, password='qazwsx')

# 负责下载文件以及将记录写入到数据库
if __name__ == '__main__':
    while True:
        task_data = requests.get(TASK_URL, headers={'token': TOKRN}).json()
        if task_data['code'] == -1:
            logger.info(task_data['msg'])
            break
        if task_data.get("data", {}).get("count", 0) == 0:
            time.sleep(2)
            continue
        lists = task_data.get("data", {}).get("lists", [])
        for task in lists[::-1]:
            if account_col.find_one({'account': task.get('account')}):
                continue
            if not ai_avatar_record_col.find_one({'task_id': task.get('task_id')}):
                task['_id'] = task.get('task_id')
                video_url = task.get('video', {}).get('video_url', '')
                video_path = download_file(video_url, TEMPPLATE_VIDEO_PATH)
                task['video_path'] = video_path
                if task.get('voice_url', '') != '':
                    voice_path = download_file(task.get('voice_url', ''), TMP_DIR)
                    task['voice_path'] = voice_path
                    task["status"] = 1
                ai_avatar_record_col.insert_one(task)
        time.sleep(2)
