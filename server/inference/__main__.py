if __name__ == "__main__":
    from server.inference.inference import Inference
    from server.utils import get_mongo_client
    from server.const import AI_AVATAR_RECORD, SUB_INFERENCE_TASK, RESTORE_QUEUE_NAME, AFFINE_TRANSFORM_CACHE_COL
    import redis

    mlient = get_mongo_client()
    db = mlient['ai_avatar']
    ai_avatar_record_col = db[AI_AVATAR_RECORD]
    affine_col = db[AFFINE_TRANSFORM_CACHE_COL]
    r = redis.Redis(host='localhost', port=6379, db=0, password='qazwsx')
    at = Inference(r, SUB_INFERENCE_TASK, 3, RESTORE_QUEUE_NAME,affine_col)
    at.run()
