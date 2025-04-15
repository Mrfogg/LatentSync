if __name__ == "__main__":
    from server.affine_train.affine_train import AffineTrainMaster
    import redis
    from server.const import AFFINE_TRANSFORM_QUEUE
    from server.const import AFFINE_TRANSFORM_CACHE_COL,AFFINE_TRAIN_CACHE_COL
    from server.utils import get_mongo_client

    mlient = get_mongo_client()
    db = mlient['ai_avatar']
    affine_col = db[AFFINE_TRANSFORM_CACHE_COL]
    affine_train_col = db[AFFINE_TRAIN_CACHE_COL]
    r = redis.Redis(host='localhost', port=6379, db=0, password='qazwsx')
    m = AffineTrainMaster(r, AFFINE_TRANSFORM_QUEUE, affine_col,affine_train_col)
    m.run()
