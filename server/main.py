import time

from server.generate_video import generate_video_u
from server.queue import Queue, AiAvatar

while True:

    ai = AiAvatar.select().where(AiAvatar.status == '-1').first()
    if ai:
        ai.status = '0'
        ai.save()
        que = Queue(ai.id)
        result_video_url = generate_video_u(ai.file_id, ai.result_audio_url, que)
        ai.result_video_url = result_video_url
        ai.status = '1'
        ai.save()
    else:
        time.sleep(1)

db.close()
