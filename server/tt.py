import numpy as np
from loguru import logger
from latentsync.utils.util import write_video

# 假设 4 个文件，每个形状为 (N, ...)，合并后 (4N, ...)
file_paths = ["temp_inf/67f894084fdb9_0.npy", "temp_inf/67f894084fdb9_60.npy", "temp_inf/67f894084fdb9_120.npy",
              "temp_inf/67f894084fdb9_180.npy"]
output_path = "merged_output.npy"
a = []
logger.info("开始")
for f in file_paths:
    t = np.load(f)
    a.append(t)
np.concatenate(a)
logger.info("结束")
# logger.info(f"开始处理")
# # 1. 获取合并后的总大小（示例：假设所有文件形状一致）
# sample_data = np.load(file_paths[0], mmap_mode='r')  # 仅读取元数据
# dtype = sample_data.dtype
# shape = sample_data.shape
# total_samples = sum(np.load(f, mmap_mode='r').shape[0] for f in file_paths)
# final_shape = (total_samples, *shape[1:])
#
# logger.info(f"中间看时间")
# # 2. 预分配输出文件（使用 memmap 避免内存爆炸）
# merged = np.memmap(output_path, dtype=dtype, mode='w+', shape=final_shape)
#
# logger.info("分块合并")
# # 3. 分块写入（避免内存爆满）
# current_idx = 0
# for file in file_paths:
#     logger.info(f"读{file}")
#     chunk = np.load(file, mmap_mode='r')  # 内存映射模式加载
#     logger.info(f"读结束{file}")
#     merged[current_idx: current_idx + len(chunk)] = chunk
#     current_idx += len(chunk)
#     del chunk  # 及时释放内存
#     logger.info("循环一次")
#
# logger.info("开始写视频")
# write_video("test_hb.mp4", merged, fps=25)
# logger.info(f"结束")
