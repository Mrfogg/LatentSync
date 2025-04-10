if __name__ == "__main__":
    from server.master.master import Master
    import redis
    from server.const import AFFINE_TRANSFORM_QUEUE
    from server.const import AI_AVATAR_RECORD
    from server.utils import get_mongo_client
    mlient = get_mongo_client()
    db = mlient['ai_avatar']
    ai_avatar_record_col = db[AI_AVATAR_RECORD]
    r = redis.Redis(host='localhost', port=6379, db=0, password='qazwsx')
    m = Master(r, ai_avatar_record_col, AFFINE_TRANSFORM_QUEUE)
    m.run()
