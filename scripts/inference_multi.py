# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from omegaconf import OmegaConf
from latentsync.pipelines_multi.lipsync_pipeline_multi import PipelineMaster

pm = None
def main(config, args, frame_callback=None):
    global pm
    import time
    if pm is None:
        pm = PipelineMaster(config, args) #
    pm.process_video(video_path=args.video_path,
                     audio_path=args.audio_path,video_out_path=args.video_out_path)

    # for batch_idx, batch in enumerate(dataloader):
    #     # ... 处理批次 ...
    #
    #     # 处理完每一帧后调用回调函数
    #     if frame_callback is not None:
    #         for frame in processed_frames:
    #             frame_callback(frame)
    #
    #     # ... 其他处理 ...
    #
    # return result
