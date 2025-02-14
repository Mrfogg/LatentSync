from kafka import KafkaConsumer, KafkaProducer
import json
from omegaconf import OmegaConf
import uuid
import os
from scripts.inference import main

# 创建 KafkaConsumer 实例
consumer = KafkaConsumer(
    'digital_suhuaren',  # 要消费的主题名称
    bootstrap_servers='localhost:9092',  # Kafka 服务地址
    auto_offset_reset='latest',  # 从最早的消息开始消费
    enable_auto_commit=True,  # 自动提交偏移量
    group_id='your_group_id',  # 消费者组ID
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',  # Kafka 服务地址
    value_serializer=lambda v: json.dumps(v).encode('utf-8')  # 消息序列化器，将消息转换为 JSON 格式
)

from dataclasses import dataclass


@dataclass
class Para:
    pass


args = Para()
args.unet_config_path = "configs/unet/second_stage.yaml"
args.inference_ckpt_path = "checkpoints/latentsync_unet.pt"
args.inference_steps = 60
args.guidance_scale = 0.05
args.seed = 1
config = OmegaConf.load(args.unet_config_path)
# 读取消息
for msg in consumer:
    consumer.commit()
    print(msg)
    # continue
    msg = msg.value
    # print(msg)
    #
    # continue
    video_output_filename = uuid.uuid4().__str__()
    args.video_path = '../assets/数字人/%s/%s' % (msg['digital_name'], msg['video_file_name'])
    args.audio_path = "../assets/tmp/" + msg['audio_name'] + ".wav"
    args.video_out_path = "../assets/tmp/" + video_output_filename + ".mp4"
    main(config, args)
    if os.path.exists(args.video_out_path):
        msg['video_file_name'] = video_output_filename
    producer.send('digital_human_result', msg)  # 发送消息到指定主题)
    producer.flush()
