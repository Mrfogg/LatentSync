import redis
import json
from omegaconf import OmegaConf
import time
import uuid
from scripts.inference_tmp import main
from dataclasses import dataclass


@dataclass
class Para:
    pass


class Queue:
    def __init__(self, rdb, msg):
        self.rdb = rdb
        self.msg = msg

    def send_complete(self, complete):
        self.msg['complete'] = str(complete)
        r.rpush('digital_human_process_result', json.dumps(self.msg))


r = redis.Redis(host='localhost', port=6379, db=0, password='qazwsx')
args = Para()
args.unet_config_path = "configs/unet/second_stage_prod.yaml"
args.inference_ckpt_path = "/data/model_test/checkpoint-4000.pt"
args.seed = 1
config = OmegaConf.load(args.unet_config_path)

while True:
    item = r.lpop('digital_human_process')
    if item is None:
        time.sleep(1)
        continue
    m = json.loads(item)
    # print(m)
    # continue
    que = Queue(rdb=r, msg=m)
    args.video_path = m['video_file_path']
    args.audio_path = m['audio_file_path']
    args.video_out_path = m['output_file_path']
    args.guidance_scale = float(m["guidance_scale"])
    args.inference_steps = int(m["inference_steps"])
    args.gpu_id = m["gpu_id"]
    main(config, args, que)
    m["complete"] = "-1"  # -1代表完全结束
    r.rpush('digital_human_process_result', json.dumps(m))
