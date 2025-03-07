from peewee import MySQLDatabase, Model, CharField, IntegerField

db = MySQLDatabase('192_144_152_86', user='192_144_152_86', password='H2sbGjXZJ3F7KZkY',
                   host='192.144.152.86', port=3306)


class Queue:
    def __init__(self, id):
        self.id = id

    def send_complete(self, complete_time):
        ai = AiAvatar.select().where(AiAvatar.id == self.id).first()
        ai.complete_time = complete_time
        ai.save()


# 定义模型类，对应数据库中的表
class AiAvatar(Model):
    id = IntegerField(primary_key=True)
    content = CharField()
    status = CharField()
    voice_name = CharField()
    delete_time = IntegerField()
    result_audio_url = CharField()
    result_video_url = CharField()
    file_id = IntegerField()
    completion_time = CharField()

    class Meta:
        database = db
        table_name = '192_tenant_ai_avatar_record'  # 表名


# 连接到数据库
db.connect()
