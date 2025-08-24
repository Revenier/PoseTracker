import redis
import os
import pickle
import hashlib
import time
from dotenv import load_dotenv, find_dotenv

# redis_host = os.getenv('REDIS_HOST', 'redis')
# redis_port = int(os.getenv('REDIS_PORT', 6379))

# r = redis.Redis(host=redis_host, port=redis_port, db=0)

load_dotenv(find_dotenv())


# Setup Redis client
r1 = redis.Redis(
    host=os.getenv("REDIS_HOST_1"),
    port=int(os.getenv("REDIS_PORT_1")),
    username=os.getenv("REDIS_USER_1"),
    password=os.getenv("REDIS_PASS_1"),
    decode_responses=False,
)

r2 = redis.Redis(
    host=os.getenv("REDIS_HOST_2"),
    port=int(os.getenv("REDIS_PORT_2")),
    username=os.getenv("REDIS_USER_2"),
    password=os.getenv("REDIS_PASS_2"),
    decode_responses=False,
)

print("✅ Connected to Redis Cloud:", r1.ping())
print("✅ Connected to Redis Cloud:", r2.ping())

def get_or_cache_result(args, processing_func):
    posture, input_data = args
    data_type = 'landmark' if processing_func.__name__ == 'landmark_logic' else 'angle'

    # Pilih Redis client berdasarkan jenis data
    redis_client = r1 if data_type == 'landmark' else r2

    start_time = time.perf_counter()

    key_data = pickle.dumps((processing_func.__name__, input_data, args))
    cache_key = f"{posture}:{data_type}:{hashlib.sha256(key_data).hexdigest()}"

    cached_data = redis_client.get(cache_key)
    if cached_data:
        try:
            obj = pickle.loads(cached_data)
            used_memory = redis_client.info().get('used_memory', -1)
            used_human = redis_client.info().get('used_memory_human', 'unknown')

            print(f"[CACHE HIT] {processing_func.__name__} | key: {cache_key} | type: {data_type} | size: {len(cached_data)} bytes | Redis: {used_human}", flush=True)
            return obj['output']
        except Exception as e:
            print(f"[CACHE ERROR] Failed to load cache for {processing_func.__name__}: {e}", flush=True)

    result = processing_func(posture, input_data)

    try:
        cache_obj = {
            'posture': posture,
            'mediapipe': input_data,
            'output': result
        }
        redis_client.set(cache_key, pickle.dumps(cache_obj))
    except Exception as e:
        print(f"[CACHE ERROR] Failed to store result: {e}", flush=True)

    duration = time.perf_counter() - start_time
    used_memory = redis_client.info().get('used_memory', -1)
    used_human = redis_client.info().get('used_memory_human', 'unknown')

    print(f"[CACHE MISS] Stored key: {cache_key} | type: {data_type} | posture: {posture} | Redis total: {used_human} ({used_memory} bytes) | time: {duration:.6f}s", flush=True)
    return result


# def get_or_cache_result(args, processing_func):
#     posture, input_data = args
#     data_type = 'landmark' if processing_func.__name__ == 'landmark_logic' else 'angle'

#     start_time = time.perf_counter()

#     key_data = pickle.dumps((processing_func.__name__, input_data, args))
#     cache_key = f"{posture}:{data_type}:{hashlib.sha256(key_data).hexdigest()}"

#     cached_data = r.get(cache_key)
#     if cached_data:
#         try:
#             obj = pickle.loads(cached_data)
#             used_memory = r.info().get('used_memory', -1)
#             used_human = r.info().get('used_memory_human', 'unknown')

#             print(f"[CACHE HIT] {processing_func.__name__} | key: {cache_key} | type: {data_type} | size: {len(cached_data)} bytes | Redis: {used_human}", flush=True)
#             return obj['output']
#         except Exception as e:
#             print(f"[CACHE ERROR] Failed to load cache for {processing_func.__name__}: {e}", flush=True)

#     result = processing_func(posture, input_data)

#     try:
#         cache_obj = {
#             'posture': posture,
#             'mediapipe': input_data,
#             'output': result
#         }
#         r.set(cache_key, pickle.dumps(cache_obj))
#     except Exception as e:
#         print(f"[CACHE ERROR] Failed to store result: {e}", flush=True)

#     duration = time.perf_counter() - start_time
#     used_memory = r.info().get('used_memory', -1)
#     used_human = r.info().get('used_memory_human', 'unknown')

#     print(f"[CACHE MISS] Stored key: {cache_key} | type: {data_type} | posture: {posture} | Redis total: {used_human} ({used_memory} bytes) | time: {duration:.6f}s", flush=True)
#     return result