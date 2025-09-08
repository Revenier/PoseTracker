import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from pathlib import Path

# Lokasi file script (…/backend/app/split_csv.py)
HERE = Path(__file__).resolve()
BACKEND_ROOT = HERE.parents[1]   # …/backend
PROJECT_ROOT = BACKEND_ROOT.parent  # root (yang ada folder backend/ dan data/)

DEFAULT_DATA_DIR = str(PROJECT_ROOT / "data")
DEFAULT_OUT_DIR  = str(PROJECT_ROOT / "data_per_pose")

# ---------- Alignment (mengacu create_pickle2.py) ----------
def align_pose_full(landmarks33x3: np.ndarray) -> np.ndarray:
    lm = landmarks33x3.copy()
    head = lm[0][:2]
    xy = lm[:, :2] - head

    # pastikan left_shoulder (11) ada di kiri right_shoulder (12)
    if xy[11][0] > xy[12][0]:
        xy[:, 0] *= -1

    # rotasi agar vektor bahu->pinggul vertikal
    shoulder_center = (xy[11] + xy[12]) / 2
    hip_center      = (xy[23] + xy[24]) / 2
    body_axis = hip_center - shoulder_center
    angle = np.arctan2(body_axis[0], body_axis[1])  # swap x/y
    R = np.array([[np.cos(-angle), -np.sin(-angle)],
                  [np.sin(-angle),  np.cos(-angle)]])
    xy = xy @ R.T

    # skala jarak bahu = 1
    dist = np.linalg.norm(xy[11] - xy[12])
    if dist > 0:
        xy /= dist

    # z distandarkan
    z = lm[:, 2:3]
    z = (z - np.mean(z)) / (np.std(z) + 1e-8)

    return np.hstack([xy, z])  # (33,3)

def make_landmark_headers():
    # 33 titik * (x,y,z) = 99 kolom
    cols = []
    for i in range(33):
        cols += [f"l{i}_x", f"l{i}_y", f"l{i}_z"]
    return cols

def main(
    data_dir: str,
    out_dir: str,
    do_processed: bool,
    minmax_angles: bool,
):
    angles_csv     = os.path.join(data_dir, "angles.csv")
    landmarks_csv  = os.path.join(data_dir, "landmarks.csv")
    labels_csv     = os.path.join(data_dir, "labels.csv")

    angles_df     = pd.read_csv(angles_csv)
    landmarks_df  = pd.read_csv(landmarks_csv)
    labels_df     = pd.read_csv(labels_csv)

    # daftar pose (seperti di labels.csv kolom 'class')
    poses = sorted(labels_df["class"].unique())

    # siapkan output dir
    os.makedirs(out_dir, exist_ok=True)

    # untuk header landmarks processed
    lm_headers = make_landmark_headers()

    print(f"Input  : {data_dir}")
    print(f"Output : {out_dir}")
    print(f"Mode   : {'PROCESSED (align+scale)' if do_processed else 'RAW (tanpa align/scale)'}")
    if do_processed and minmax_angles:
        print(f"Angles : MinMaxScaler feature-wise (0..1)")
    print()

    for pose in poses:
        vid_ids = labels_df.loc[labels_df["class"] == pose, "vid_id"].astype(int).tolist()
        if not vid_ids:
            print(f"[SKIP] {pose}: tidak ada vid_id.")
            continue

        # ---------- LANDMARKS ----------
        lms_pose = landmarks_df[landmarks_df["vid_id"].astype(int).isin(vid_ids)].copy()
        # simpan raw (opsional)
        lms_raw_path = os.path.join(out_dir, f"{pose}_landmarks_raw.csv")

        # processed: drop meta, align, flatten
        lms_keep_cols = [c for c in lms_pose.columns if c not in ("vid_id", "frame_order")]
        lms_arr = lms_pose[lms_keep_cols].to_numpy(dtype=float)  # (N,99)

        if do_processed:
            aligned_rows = []
            for row in lms_arr:
                lm = row.reshape(33, 3)
                aligned = align_pose_full(lm).flatten()  # (99,)
                aligned_rows.append(aligned)
            lms_processed = pd.DataFrame(aligned_rows, columns=lm_headers)
            lms_out_path = os.path.join(out_dir, f"{pose}_landmarks_processed.csv")
            lms_processed.to_csv(lms_out_path, index=False)
            print(f"[OK] Landmarks {pose}: processed → {os.path.relpath(lms_out_path)}")
        else:
            # RAW: tulis apa adanya (tetap drop meta biar bersih)
            lms_raw = pd.DataFrame(lms_arr)
            lms_raw.to_csv(lms_raw_path, index=False)
            print(f"[OK] Landmarks {pose}: raw → {os.path.relpath(lms_raw_path)}")

        # ---------- ANGLES ----------
        ang_pose = angles_df[angles_df["vid_id"].astype(int).isin(vid_ids)].copy()
        ang_keep_cols = [c for c in ang_pose.columns if c not in ("vid_id", "frame_order")]
        ang_arr = ang_pose[ang_keep_cols].to_numpy(dtype=float)

        if do_processed and ang_arr.size > 0 and minmax_angles:
            scaler = MinMaxScaler()
            ang_arr = scaler.fit_transform(ang_arr)

        ang_out = pd.DataFrame(ang_arr, columns=[f"a{i}" for i in range(ang_arr.shape[1])])
        ang_out_path = os.path.join(out_dir, f"{pose}_angles_{'processed' if (do_processed and minmax_angles) else 'raw'}.csv")
        ang_out.to_csv(ang_out_path, index=False)
        print(f"[OK] Angles {pose}: {'processed' if (do_processed and minmax_angles) else 'raw'} → {os.path.relpath(ang_out_path)}")

    print("\nSelesai.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split master CSV into per-pose CSVs (raw or processed).")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Folder berisi angles.csv, landmarks.csv, labels.csv")
    parser.add_argument("--out-dir",  default=DEFAULT_OUT_DIR,  help="Folder output per-pose")
    parser.add_argument("--processed", action="store_true", help="Kalau aktif: landmarks di-align & z-normalize, angles MinMax")
    parser.add_argument("--minmax-angles", action="store_true", help="Kalau --processed aktif, aktifkan MinMax untuk angles")
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        do_processed=args.processed,
        minmax_angles=args.minmax_angles,
    )
