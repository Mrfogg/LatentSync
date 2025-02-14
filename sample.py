import os
import random
import shutil


def split_files(source_folder, train_folder, val_folder, train_ratio=0.8):
    """
    将源文件夹中的文件随机分成两部分，分别存储到训练文件夹和验证文件夹中。

    :param source_folder: 源文件夹路径
    :param train_folder: 训练文件夹路径
    :param val_folder: 验证文件夹路径
    :param train_ratio: 训练集占总文件的比例，默认为0.8
    """
    # 确保目标文件夹存在
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # 获取源文件夹中的所有文件
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    random.shuffle(files)  # 随机打乱文件顺序

    # 计算训练集和验证集的文件数量
    train_count = int(len(files) * train_ratio)
    val_count = len(files) - train_count

    print(f"Total files: {len(files)}")
    print(f"Train files: {train_count}")
    print(f"Val files: {val_count}")

    # 分配文件到训练集和验证集
    for i, file in enumerate(files):
        source_path = os.path.join(source_folder, file)
        if i < train_count:
            target_path = os.path.join(train_folder, file)
        else:
            target_path = os.path.join(val_folder, file)

        # 复制文件到目标文件夹
        shutil.copy2(source_path, target_path)
        print(f"Copied {source_path} to {target_path}")

    print("File splitting complete.")


# 使用示例
source_folder = "high_visual_quality"  # 源文件夹路径
train_folder = "high_visual_quality/train"  # 训练文件夹路径
val_folder = "high_visual_quality/val"  # 验证文件夹路径

split_files(source_folder, train_folder, val_folder, train_ratio=0.8)