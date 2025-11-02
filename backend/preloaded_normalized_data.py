import argparse
import hashlib
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import redis
from sklearn.preprocessing import normalize

# Import align_landmarks from logic.py
from app.logic import align_landmarks

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB", "0"))

def get_redis():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
    r.ping()
    return r

def read_numeric_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path, header=0)
    df = df.select_dtypes(include=["number"])
    return df.to_numpy(dtype=float)

def row_id_hex(function_name: str, posture: str, row: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(function_name.encode("utf-8"))
    h.update(pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL))
    h.update(posture.encode("utf-8"))
    return h.hexdigest()

def normalize_and_align(ref_landmarks: np.ndarray) -> np.ndarray:
    """Normalize and align all landmark frames."""

    aligned_refs = []
    for ref in ref_landmarks:
        ref_pose = ref.reshape((33, 3))
        aligned_ref = align_landmarks(ref_pose)
        aligned_refs.append(aligned_ref.flatten())
    aligned_refs = np.array(aligned_refs)
    ref_norm = normalize(aligned_refs, axis=1)
    return ref_norm

def save_rows(r, posture, data_type, function_name, rows):  
    new_cnt = 0
    found_cnt = 0

    for idx, row in enumerate(rows):
        key = f"{posture}:{data_type}:{row_id_hex(function_name, posture, row)}"
        value = pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL)

        inserted = r.set(key, value, nx=True)
        if inserted:
            new_cnt += 1
            status = "New"
        else:
            found_cnt += 1
            status = "Found"

        print(f"[{status}] {data_type} row {idx} -> {key}")

    return new_cnt, found_cnt, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        default=str(Path(__file__).parent / "app" / "data_per_pose"),
        help="Folder data_per_pose (default: <script_dir>/app/data_per_pose)",
    )
    args = ap.parse_args()
 
    data_dir = Path(args.dir)
    candidates = [
        data_dir,
        Path(__file__).parent / "app" / "data_per_pose",
        Path(__file__).parent / "data_per_pose",
        Path("/app/data_per_pose"),
        Path("/data_per_pose"),
    ]
    data_dir = next((p for p in candidates if p.exists()), None)
    if data_dir is None:
        print("❌ Folder data_per_pose tidak ditemukan di salah satu lokasi kandidat:")
        for p in candidates:
            print(" -", p)
        sys.exit(1)
    else:
        print(f"📁 data_per_pose digunakan: {data_dir}")

    r = get_redis()
    print(f"✅ Connected to Redis ({REDIS_HOST}:{REDIS_PORT}, db={REDIS_DB})")

    poses = ["jumping_jack", "push_up", "situp", "squat"]

    total_new = 0
    total_found = 0
    total_rows = 0

    for posture in poses:
        print(f"\n=== 🏋️ Normalizing + Uploading posture: {posture} ===")

        lm_path = data_dir / f"{posture}_landmarks_raw.csv"
        if not lm_path.is_file():
            print(f"❌ File tidak ditemukan: {lm_path}")
            continue

        # Load, normalize, align, normalize again
        landmarks = read_numeric_csv(lm_path)
        normalized_landmarks = normalize_and_align(landmarks)

        lm_new, lm_found, lm_total = save_rows(r, posture, "landmark", "landmark_logic", normalized_landmarks)
        print(f"🔎 Landmarks: total_rows={lm_total}, new={lm_new}, found={lm_found}")

        total_new += lm_new
        total_found += lm_found
        total_rows += lm_total

        an_path = data_dir / f"{posture}_angles_raw.csv"
        if an_path.is_file():
            angles = read_numeric_csv(an_path)
            # Just normalize angles feature-wise
            an_new, an_found, an_total = save_rows(r, posture, "angle", "angle_logic", angles)
            print(f"🔎 Angles   : total_rows={an_total}, new={an_new}, found={an_found}")

            total_new += an_new
            total_found += an_found
            total_rows += an_total
        else:
            print("ℹ️  Angles file tidak ditemukan — dilewati.")

    print("\n✅ Selesai preload data normalized ke Redis.")
    print(f"📊 TOTAL KESELURUHAN → total_rows={total_rows}, new={total_new}, found_duplicate={total_found}")


if __name__ == "__main__":
    main()
