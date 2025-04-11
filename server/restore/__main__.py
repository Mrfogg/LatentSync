if __name__ == "__main__":
    from server.restore.restore import Restore
    from server.utils import get_mongo_client
    from server.const import RESTORE_QUEUE_NAME, AFFINE_TRANSFORM_CACHE_COL, AI_AVATAR_RECORD
    import redis

    mlient = get_mongo_client()
    db = mlient['ai_avatar']
    r = redis.Redis(host='localhost', port=6379, db=0, password='qazwsx')
    affine_col = db[AFFINE_TRANSFORM_CACHE_COL]
    ai_avatar_record_col = db[AI_AVATAR_RECORD]
    rt = Restore(r, RESTORE_QUEUE_NAME, affine_col, ai_avatar_record_col)
    rt.run()
