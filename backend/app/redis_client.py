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

    while True:
        # Ambil batch key dari Redis sesuai pola
        cursor, keys = r.scan(cursor=cursor, match=pattern, count=1000)

        # Pastikan semua key jadi string dan hanya ambil yang prefix-nya cocok
        clean_keys = []
        for k in keys:
            if isinstance(k, bytes):
                k = k.decode()
            if k.startswith(f"{posture}:{data_type}:"):
                clean_keys.append(k)

        # Kalau ada key yang valid, ambil semua valuenya sekaligus
        if clean_keys:
            vals = r.mget(clean_keys)
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
                    row = np.asarray(arr, dtype=dtype)
                    if row.ndim == 1:
                        rows.append(row.tolist())
                except Exception as e:
                    print(f"[Decode Error] {e}", flush=True)
                    continue

        # Kalau cursor == 0 berarti sudah selesai scan semua key
        if cursor == 0:
            break

    if not rows:
        print(f"[Redis] ❌ No rows found for {posture}:{data_type}", flush=True)
        return np.empty((0, 0), dtype=dtype)

    print(f"[Redis] ✅ Loaded {len(rows)} rows for {posture}:{data_type}", flush=True)
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