import os

# 文件路径
file_path = "uploads/video/20250223/202502231134452bdbe1839.mp4"

# 提取文件名
file_name = os.path.basename(file_path)
print(file_name)