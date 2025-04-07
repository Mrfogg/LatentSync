import requests

token = 'xUSdpIaKdgaAUQzh'

REQ_URL = 'https://hfw-api.lingverse.co/api/v1/hifly/task/create'
STATUS_URL = 'https://hfw-api.lingverse.co/api/v1/hifly/task/inspect'
# r = redis.Redis(host='localhost', port=6379, db=0, password='qazwsx')


def generate_video_task(video_url, audio_url):
    res = requests.post(REQ_URL, headers={'Authorization': 'Bearer ' + token, 'content-type': 'application/json'},
                        json={
                            'audio_url': audio_url,
                            'video_url': video_url,
                            'only_generate_audio': 0
                        })
    print(res.json())
    return res.json().get('job_id')


def query_task(job_id: int):
    res = requests.post(STATUS_URL, headers={'Authorization': 'Bearer ' + token, 'content-type': 'application/json'},
                        json={
                            'job_id': job_id
                        })
    return res.json().get('video_Url', ''), res.json().get('status', 0)


# job_id = generate_video_task('http://192.144.152.86/uploads/video/20250309/20250309194457ddff78503.mp4',
#                              'http://192.144.152.86/uploads/file/20250324/20250324105005b31203100.wav')
# video = query_task(4835517)
# print(video)
