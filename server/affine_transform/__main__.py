if __name__ == "__main__":
    from server.affine_transform.affine_transform import AffineTransform
    from server.utils import get_mongo_client
    from server.const import AI_AVATAR_RECORD, AFFINE_TRANSFORM_QUEUE, AFFINE_TRANSFORM_CACHE_COL
    import redis

    mlient = get_mongo_client()
    db = mlient['ai_avatar']
    ai_avatar_record_col = db[AI_AVATAR_RECORD]
    affine_col = db[AFFINE_TRANSFORM_CACHE_COL]
    r = redis.Redis(host='localhost', port=6379, db=0, password='qazwsx')
    at = AffineTransform(r, AFFINE_TRANSFORM_QUEUE, affine_col)
    at.run()
