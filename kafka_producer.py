from kafka import KafkaProducer
import json

# 创建 KafkaProducer 实例
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',  # Kafka 服务地址
    value_serializer=lambda v: json.dumps(v).encode('utf-8')  # 消息序列化器，将消息转换为 JSON 格式
)

# 写入消息
message = {"key": "value"}  # 消息内容
producer.send('test', message)  # 发送消息到指定主题

# 等待所有消息发送完成
producer.flush()

print("Message sent successfully!")