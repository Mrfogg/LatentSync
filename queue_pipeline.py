import redis
import json
from omegaconf import OmegaConf
import time
import uuid
import torch
from latentsync.pipelines_multi.lipsync_pipeline_multi import PipelineMaster
from dataclasses import dataclass

#
@dataclass
class Para:
    pass


# class Queue:
#     def __init__(self, rdb, msg):
#         self.rdb = rdb
#         self.msg = msg
#
#     def send_complete(self, complete):
#         self.msg['complete'] = str(complete)
#         self.rdb.rpush('digital_human_process_result', json.dumps(self.msg))

if __name__ == '__main__':
    r = redis.Redis(host='localhost', port=6379, db=0, password='qazwsx')
    args = Para()
    args.unet_config_path = "configs/unet/second_stage_prod.yaml"
    args.inference_ckpt_path = "/home/qc/data/model/checkpoint-31000.pt"
    args.seed = 1
    config = OmegaConf.load(args.unet_config_path)
    pm = PipelineMaster(config, args)  #
    print("start")
    while True:
        torch.cuda.empty_cache()
        item = r.lpop('digital_human_process')
        if item is None:
            time.sleep(1)
            continue
        m = json.loads(item)
        # print(m)
        # continue
        args.video_path = m['video_file_path']
        args.audio_path = m['audio_file_path']
        args.video_out_path = m['output_file_path']
        args.gpu_id = m["gpu_id"]
        pm.process_video(m['video_file_path'], m['audio_file_path'], m['output_file_path'])
        m["complete"] = "-1"  # -1代表完全结束
        r.rpush('digital_human_process_result', json.dumps(m))
