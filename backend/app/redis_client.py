import redis
import os
import pickle
import hashlib
import time

# Ambil host dari environment variable
redis_host = os.getenv('REDIS_HOST', 'redis')
redis_port = int(os.getenv('REDIS_PORT', 6379))

r = redis.Redis(host=redis_host, port=redis_port, db=0)


def get_or_cache_result(input_data, processing_func, *args):
    """
    Enhanced: Track both cache hit and miss size (from Redis or local pickle), plus memory info.
    """
    key_data = pickle.dumps((processing_func.__name__, input_data, args))
    cache_key = hashlib.sha256(key_data).hexdigest()

    start_time = time.perf_counter()
    cached = r.get(cache_key)

    if cached:
        result = pickle.loads(cached)
        size_in_bytes = len(cached)
        duration = time.perf_counter() - start_time

        print(f"[CACHE HIT] {processing_func.__name__} | key: {cache_key[:8]}... | size: {size_in_bytes} bytes | time: {duration:.6f}s")
        return result

    # cache miss
    print(f"[CACHE MISS] Running {processing_func.__name__} with new data.")
    result = processing_func(*input_data, *args)
    result_pickle = pickle.dumps(result)
    size_in_bytes = len(result_pickle)

    r.set(cache_key, result_pickle)

    try:
        used_memory = int(r.info()['used_memory'])
        used_human = r.info()['used_memory_human']
    except:
        used_memory = -1
        used_human = 'unknown'

    duration = time.perf_counter() - start_time
    print(f"[CACHE MISS] Stored key: {cache_key[:8]}... | size: {size_in_bytes} bytes | Redis total: {used_human} ({used_memory} bytes) | time: {duration:.6f}s")

    return result