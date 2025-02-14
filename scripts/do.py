from tqdm import tqdm
import time

# 使用 tqdm 包装 range
for i in tqdm(range(100)):
    time.sleep(0.1)  # 模拟耗时操作