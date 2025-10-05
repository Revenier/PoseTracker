# import redis
# import os
# import pickle
# import hashlib
# import time
# import numpy as np, time


# """" Versi test di postman satu landmark """

# REDIS_HOST = os.getenv("REDIS_HOST", "redis")
# REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
# REDIS_DB   = int(os.getenv("REDIS_DB", "0"))

# r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)

# try:
#     pong = r.ping()
#     print(f"✅ Connected to local Redis ({REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}):", pong)
# except Exception as e:
#     print("❌ Failed to connect to local Redis:", e)


# def _build_cache_key(posture: str, data_type: str, processing_func_name: str, input_data: np.ndarray):
#     """
#     Bangun cache key deterministik seperti versi cloud:
#     {posture}:{data_type}:{sha256(function_name + input + posture)}
#     """
#     # Hati-hati: gunakan bentuk numpy array agar stabil
#     arr = np.asarray(input_data, dtype=float)
#     key_blob = pickle.dumps((processing_func_name, posture, arr.shape, arr))
#     return f"{posture}:{data_type}:{hashlib.sha256(key_blob).hexdigest()}"


# def get_or_cache_result(args, processing_func):
#     """
#     1) Cek cache di Redis lokal
#     2) Kalau tidak ada -> proses, simpan ke Redis
#     3) Kembalikan output
#     Flow (penamaan key & log) mengikuti versi Cloud agar konsisten.
#     """
#     posture, input_data = args

#     # Tentukan data_type sesuai flow Cloud
#     data_type = "landmark" if processing_func.__name__ == "landmark_logic" else "angle"

#     # Siapkan key
#     arr = np.asarray(input_data, dtype=float)
#     cache_key = _build_cache_key(posture, data_type, processing_func.__name__, arr)

#     # --- Cache HIT ---
#     cached = r.get(cache_key)
#     if cached:
#         try:
#             # TODO (DONE): tolong cek ini beneran load dr pickle ga. 
#             # NOTE: Redis_client tidak membaca file .pkl langsung, hanya memuat data pickle-serialized dari Redis.
#             # 1️⃣ Redis_client tidak membaca file .pkl dari folder pickle/
#             # 2️⃣ Data diambil langsung dari Redis dan didecode pakai pickle.loads()
#             # 3️⃣ Pickle di sini hanya format penyimpanan di Redis, bukan sumber file .pkl
#             obj = pickle.loads(cached)
#             used_human = r.info().get("used_memory_human", "unknown")
#             print(
#                 f"[CACHE HIT] {processing_func.__name__} | key: {cache_key} "
#                 f"| type: {data_type} | size: {len(cached)} bytes | Redis: {used_human}",
#                 flush=True,
#             )
#             return obj["output"]
#         except Exception as e:
#             print(f"[CACHE ERROR] Failed to load cache for {processing_func.__name__}: {e}", flush=True)

#     # --- Cache MISS -> proses & simpan ---
#     t0 = time.perf_counter()
#     # Proses data by logic function
#     result = processing_func(posture, arr) 

#     try:
#         cache_obj = {"posture": posture, "mediapipe": arr, "output": result}
#         r.set(cache_key, pickle.dumps(cache_obj))
#     except Exception as e:
#         print(f"[CACHE ERROR] Failed to store result: {e}", flush=True)

#     dt = time.perf_counter() - t0
#     info = r.info()
#     used_human = info.get("used_memory_human", "unknown")
#     used_bytes = info.get("used_memory", -1)

#     print(
#         f"[CACHE MISS] Stored key: {cache_key} | type: {data_type} | posture: {posture} "
#         f"| Redis total: {used_human} ({used_bytes} bytes) | time: {dt:.6f}s",
#         flush=True,
#     )
#     return result


# def get_or_cache_result_batch(posture, samples, processing_func, *, log_summary=True):
#     """
#     Proses batch (list of 99-float arrays) dengan flow yang sama:
#     - Tetap pakai get_or_cache_result() per item (supaya HIT/MISS per-sample tercatat)
#     - Akhirnya, cetak ringkasan [BATCH]
#     """
#     if isinstance(samples, (list, tuple)) and samples and isinstance(samples[0], (list, tuple, np.ndarray)):
#         batch = [np.asarray(s, dtype=float) for s in samples]
#     else:
#         batch = [np.asarray(samples, dtype=float)]

#     t0 = time.perf_counter()
#     outs = []
#     for arr in batch:
#         outs.append(get_or_cache_result((posture, arr), processing_func))

#     if log_summary:
#         dt = time.perf_counter() - t0
#         print(
#             f"[BATCH] {processing_func.__name__} | posture:{posture} | items:{len(batch)} | time:{dt:.6f}s",
#             flush=True,
#         )
#     return outs

"""
Redis client (READ-ONLY) untuk proyek motion detection.

 data di Redis:
- Disimpan per-baris  (tanpa vid_id & frame_id).
- Key pattern: "{posture}:{data_type}:{hexid}"
  contoh: "push_up:landmark:260a28e7..." atau "push_up:angle:f9322dc6..."

Tujuan file ini:
- Menyediakan fungsi pembacaan matriks fitur untuk logic.py:
    load_feature_matrix(posture, data_type) -> np.ndarray shape (N, D)
- Tidak ada penulisan ke Redis (read-only).
"""

from __future__ import annotations

import os
from typing import Literal, List
import numpy as np
import redis
import pickle

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB", "0"))

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)

def load_feature_matrix(posture: str, data_type: Literal["landmark", "angle"], *, dtype=float):
    pattern = f"{posture}:{data_type}:*"
    cursor = 0
    rows = []

    # Loop untuk scan semua key yang cocok di Redis
    while True:
        cursor, keys = r.scan(cursor=cursor, match=pattern, count=1000)
        if keys:
            vals = r.mget(keys)
            for raw in vals:
                if raw is None:
                    continue
                try:
                    #NOTE: Redis_client tidak membaca file .pkl langsung, hanya memuat data pickle-serialized dari Redis.
                    # Data sudah dalam bentuk pickle di Redis.
                    # Data diambil langsung dari Redis dan didecode pakai pickle.loads()
                    # Pickle di sini hanya format penyimpanan di Redis, bukan sumber file .pkl
                    # Redis itu cuma bisa nyimpen string atau bytes butuh format biner yang bisa dibaca Redis.
                    arr = pickle.loads(raw)
                    row = np.asarray(arr, dtype=dtype).tolist()
                except Exception:
                    continue
                rows.append(row)
        
        # Kalau cursor == 0 berarti sudah selesai scan semua key
        if cursor == 0:
            print(f"[Redis] Loaded {len(rows)} rows for {posture}:{data_type}", flush=True)
            break

    # Kalau tidak ada data sama sekali, kembalikan array kosong
    if not rows:
        return np.empty((0, 0), dtype=dtype)

    return np.asarray(rows, dtype=dtype)


def count_keys(posture: str, data_type: Literal["landmark", "angle"]) -> int:
    """
    Hitung jumlah key yang cocok untuk pola {posture}:{data_type}:* (read-only).
    """
    pattern = f"{posture}:{data_type}:*"
    cursor = 0
    total = 0

    while True:
        cursor, keys = r.scan(cursor=cursor, match=pattern, count=2000)
        total += len(keys)
        if cursor == 0:
            break

    return total
