import redis
import os
import pickle
import hashlib
import time
from dotenv import load_dotenv, find_dotenv
import numpy as np, time

load_dotenv(find_dotenv())

"""" Versi test di postman satu landmark """

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB", "0"))

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)

# (Opsional) sanity check
try:
    pong = r.ping()
    print(f"✅ Connected to local Redis ({REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}):", pong)
except Exception as e:
    print("❌ Failed to connect to local Redis:", e)


def _build_cache_key(posture: str, data_type: str, processing_func_name: str, input_data: np.ndarray):
    """
    Bangun cache key deterministik seperti versi cloud:
    {posture}:{data_type}:{sha256(function_name + input + posture)}
    """
    # Hati-hati: gunakan bentuk numpy array agar stabil
    arr = np.asarray(input_data, dtype=float)
    key_blob = pickle.dumps((processing_func_name, posture, arr.shape, arr))
    return f"{posture}:{data_type}:{hashlib.sha256(key_blob).hexdigest()}"


def get_or_cache_result(args, processing_func):
    """
    1) Cek cache di Redis lokal
    2) Kalau tidak ada -> proses, simpan ke Redis
    3) Kembalikan output
    Flow (penamaan key & log) mengikuti versi Cloud agar konsisten.
    """
    posture, input_data = args

    # Tentukan data_type sesuai flow Cloud
    data_type = "landmark" if processing_func.__name__ == "landmark_logic" else "angle"

    # Siapkan key
    arr = np.asarray(input_data, dtype=float)
    cache_key = _build_cache_key(posture, data_type, processing_func.__name__, arr)

    # --- Cache HIT ---
    cached = r.get(cache_key)
    if cached:
        try:
            obj = pickle.loads(cached)
            used_human = r.info().get("used_memory_human", "unknown")
            print(
                f"[CACHE HIT] {processing_func.__name__} | key: {cache_key} "
                f"| type: {data_type} | size: {len(cached)} bytes | Redis: {used_human}",
                flush=True,
            )
            return obj["output"]
        except Exception as e:
            print(f"[CACHE ERROR] Failed to load cache for {processing_func.__name__}: {e}", flush=True)

    # --- Cache MISS -> proses & simpan ---
    t0 = time.perf_counter()
    result = processing_func(posture, arr)

    try:
        cache_obj = {"posture": posture, "mediapipe": arr, "output": result}
        r.set(cache_key, pickle.dumps(cache_obj))
    except Exception as e:
        print(f"[CACHE ERROR] Failed to store result: {e}", flush=True)

    dt = time.perf_counter() - t0
    info = r.info()
    used_human = info.get("used_memory_human", "unknown")
    used_bytes = info.get("used_memory", -1)

    print(
        f"[CACHE MISS] Stored key: {cache_key} | type: {data_type} | posture: {posture} "
        f"| Redis total: {used_human} ({used_bytes} bytes) | time: {dt:.6f}s",
        flush=True,
    )
    return result


def get_or_cache_result_batch(posture, samples, processing_func, *, log_summary=True):
    """
    Proses batch (list of 99-float arrays) dengan flow yang sama:
    - Tetap pakai get_or_cache_result() per item (supaya HIT/MISS per-sample tercatat)
    - Akhirnya, cetak ringkasan [BATCH]
    """
    if isinstance(samples, (list, tuple)) and samples and isinstance(samples[0], (list, tuple, np.ndarray)):
        batch = [np.asarray(s, dtype=float) for s in samples]
    else:
        batch = [np.asarray(samples, dtype=float)]

    t0 = time.perf_counter()
    outs = []
    for arr in batch:
        outs.append(get_or_cache_result((posture, arr), processing_func))

    if log_summary:
        dt = time.perf_counter() - t0
        print(
            f"[BATCH] {processing_func.__name__} | posture:{posture} | items:{len(batch)} | time:{dt:.6f}s",
            flush=True,
        )
    return outs