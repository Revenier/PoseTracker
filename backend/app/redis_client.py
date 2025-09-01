import redis
import os
import pickle
import hashlib
import time
from dotenv import load_dotenv, find_dotenv
import numpy as np, time

load_dotenv(find_dotenv())


""" Versi Redis Cloud """
""" Docker gk perlu dinyalain """

#Setup Redis untuk landmark
r1 = redis.Redis(
    host=os.getenv("REDIS_HOST_1"),
    port=int(os.getenv("REDIS_PORT_1")),
    username=os.getenv("REDIS_USER_1"),
    password=os.getenv("REDIS_PASS_1"),
    decode_responses=False,  # False supaya data disimpan dalam bentuk bytes (perlu untuk pickle)
)

# Setup Redis untuk angle
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
    """
    Fungsi utama untuk:
    1. Mengecek apakah hasil perhitungan sudah ada di Redis (cache hit).
    2. Kalau tidak ada, proses input dengan fungsi (cache miss).
    3. Simpan hasilnya ke Redis untuk pemanggilan berikutnya.
    """
    posture, input_data = args

    # Tentukan tipe data berdasarkan fungsi yang dipakai
    data_type = 'landmark' if processing_func.__name__ == 'landmark_logic' else 'angle'

    # Pilih Redis client sesuai tipe data
    redis_client = r1 if data_type == 'landmark' else r2

    # Mulai stopwatch untuk durasi logging 
    start_time = time.perf_counter()

    # Buat key unik untuk Redis berdasarkan posture + tipe + hash input
    key_data = pickle.dumps((processing_func.__name__, input_data, args))
    cache_key = f"{posture}:{data_type}:{hashlib.sha256(key_data).hexdigest()}"

    # Cek apakah data sudah ada di Redis atau cache
    cached_data = redis_client.get(cache_key)
    if cached_data:
        try:
            # Decode pickle hasil cache
            obj = pickle.loads(cached_data)

            # Memory dalam bytes
            used_memory = redis_client.info().get('used_memory', -1)

            # Memory dalam format M
            used_human = redis_client.info().get('used_memory_human', 'unknown')

            # Log kalau data sudah ada di redis atau cache
            print(
                f"[CACHE HIT] {processing_func.__name__} | key: {cache_key} "
                f"| type: {data_type} | size: {len(cached_data)} bytes | Redis: {used_human}",
                flush=True
            )
            return obj['output']
        except Exception as e:
            # Kalau gagal decode, tampilkan error
            print(f"[CACHE ERROR] Failed to load cache for {processing_func.__name__}: {e}", flush=True)

    # Kalau cache miss, jalankan fungsi processing
    result = processing_func(posture, input_data)

    try:
        # Simpan hasil ke Redis dengan format dict -> pickle
        cache_obj = {
            'posture': posture,
            'mediapipe': input_data,
            'output': result
        }
        redis_client.set(cache_key, pickle.dumps(cache_obj))
    except Exception as e:
        print(f"[CACHE ERROR] Failed to store result: {e}", flush=True)

    # durasi proses dan tampilkan info memory Redis
    duration = time.perf_counter() - start_time

    # Memory dalam bytes
    used_memory = redis_client.info().get('used_memory', -1)

    # Memory dalam format M atau K
    used_human = redis_client.info().get('used_memory_human', 'unknown')

    # Log kalau data berhasil disimpan ke redis
    print(
        f"[CACHE MISS] Stored key: {cache_key} | type: {data_type} | posture: {posture} "
        f"| Redis total: {used_human} ({used_memory} bytes) | time: {duration:.6f}s",
        flush=True
    )
    return result


# Fungsi untuk proses batch dengan caching
def get_or_cache_result_batch(posture, samples, processing_func, *, log_summary=True):
    """
    Proses banyak sample (list of [99]) satu-per-satu.
    - Per item tetap pakai get_or_cache_result()
    - Akhir batch (optional) cetak 1 baris ringkasan: [BATCH] ...
    Return: list output sesuai urutan input.
    """

    # Jika 'samples' adalah list/tuple dan tidak kosong
    # dan elemen pertamanya juga list/tuple/np.ndarray (artinya: ini memang batch / list of arrays),
    if isinstance(samples, (list, tuple)) and samples and isinstance(samples[0], (list, tuple, np.ndarray)):
        batch = [np.asarray(s, dtype=float) for s in samples]   
    else:
        batch = [np.asarray(samples, dtype=float)]              

    t0 = time.perf_counter()
    outs = []
    for arr in batch:
        # panggil fungsi single-item yang sudah ada (get_or_cache_result)
        # - argumen: (posture, arr) + processing_func (mis. angle_logic / landmark_logic)
        # - fungsi ini yang akan handle Redis HIT/MISS dan nge-print log aslinya per item
        outs.append(get_or_cache_result((posture, arr), processing_func))

    if log_summary:
        dt = time.perf_counter() - t0
        print(f"[BATCH] {processing_func.__name__} | posture:{posture} | items:{len(batch)} | time:{dt:.6f}s", flush=True)

    return outs





"""" Versi Lokal / Docker """
"""" Versi test di postman satu landmark """

# redis_host = os.getenv('REDIS_HOST', 'redis')
# redis_port = int(os.getenv('REDIS_PORT', 6379))

# #Landmark dan angle di redis yang sama
# r = redis.Redis(host=redis_host, port=redis_port, db=0)

# def get_or_cache_result(args, processing_func):
#     posture, input_data = args
#     data_type = 'landmark' if processing_func.__name__ == 'landmark_logic' else 'angle'

#     # Mulai stopwatch untuk durasi logging 
#     start_time = time.perf_counter()

#     # Bikin cache key unik pakai hash dari input + function
#     key_data = pickle.dumps((processing_func.__name__, input_data, args))
#     cache_key = f"{posture}:{data_type}:{hashlib.sha256(key_data).hexdigest()}"

#     # Coba ambil dari cache atau Redis
#     cached_data = r.get(cache_key)
#     if cached_data:
#         try:
#             # Decode pickle hasil cache
#             obj = pickle.loads(cached_data)
#             # Memory dalam bytes
#             used_memory = r.info().get('used_memory', -1)

#             # Memory dalam format M
#             used_human = r.info().get('used_memory_human', 'unknown')

#             print(f"[CACHE HIT] {processing_func.__name__} | key: {cache_key} | type: {data_type} | size: {len(cached_data)} bytes | Redis: {used_human}", flush=True)
#             return obj['output']
#         except Exception as e:

#             # Kalau gagal decode, tampilkan error
#             print(f"[CACHE ERROR] Failed to load cache for {processing_func.__name__}: {e}", flush=True)

#     # Kalau belum ada di cache atau Redis maka proses data dulu
#     result = processing_func(posture, input_data)

#     try:
#         # Simpan hasil ke Redis dengan format dict -> pickle
#         cache_obj = {
#             'posture': posture,
#             'mediapipe': input_data,
#             'output': result
#         }
#         r.set(cache_key, pickle.dumps(cache_obj))
#     except Exception as e:
#         print(f"[CACHE ERROR] Failed to store result: {e}", flush=True)

#     # durasi proses dan tampilkan info memory Redis
#     duration = time.perf_counter() - start_time

#     # Memory dalam bytes
#     used_memory = r.info().get('used_memory', -1)

#     # Memory dalam format M atau K
#     used_human = r.info().get('used_memory_human', 'unknown')

#     # Log kalau data berhasil disimpan ke redis
#     print(f"[CACHE MISS] Stored key: {cache_key} | type: {data_type} | posture: {posture} | Redis total: {used_human} ({used_memory} bytes) | time: {duration:.6f}s", flush=True)
#     return result