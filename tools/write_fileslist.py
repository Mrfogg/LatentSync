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
import sys
sys.path.insert(0,"/home/qc/ai_server/ai_server/LatentSync")
from tqdm import tqdm
from latentsync.utils.util import gather_video_paths_recursively

def write_fileslist(fileslist_path):
    with open(fileslist_path, "w") as _:
        pass


def append_fileslist(fileslist_path, video_paths):
    with open(fileslist_path, "w") as f:
        for video_path in tqdm(video_paths):
            f.write(f"{video_path}\n")


def process_input_dir(fileslist_path, input_dir):
    print(f"Processing input dir: {input_dir}")
    video_paths = gather_video_paths_recursively(input_dir)
    append_fileslist(fileslist_path, video_paths)


if __name__ == "__main__":
    fileslist_path = "/home/qc/workspace/LatentSync/fileslist/val.txt"

    write_fileslist(fileslist_path)
    process_input_dir(fileslist_path, "/media/qc/1/high_visual_quality/val")

    fileslist_path = "/home/qc/workspace/LatentSync/fileslist/train.txt"

    write_fileslist(fileslist_path)
    process_input_dir(fileslist_path, "/media/qc/1/high_visual_quality/train")
