import os
import pandas as pd
from pathlib import Path

# Lokasi file script (…/backend/app/split_csv.py)
HERE = Path(__file__).resolve()
BACKEND_ROOT = HERE.parents[1]   # …/backend
PROJECT_ROOT = BACKEND_ROOT.parent  # root (yang ada folder backend/ dan data/)

DEFAULT_DATA_DIR = str(PROJECT_ROOT / "data")
DEFAULT_OUT_DIR  = str(PROJECT_ROOT / "data_per_pose")

def main(data_dir: str, out_dir: str):
    angles_csv     = os.path.join(data_dir, "angles.csv")
    landmarks_csv  = os.path.join(data_dir, "landmarks.csv")
    labels_csv     = os.path.join(data_dir, "labels.csv")

    # load semua csv
    angles_df     = pd.read_csv(angles_csv)
    landmarks_df  = pd.read_csv(landmarks_csv)
    labels_df     = pd.read_csv(labels_csv)

    # daftar pose unik (dari labels)
    poses = sorted(labels_df["class"].unique())

    os.makedirs(out_dir, exist_ok=True)
    print(f"Input  : {data_dir}")
    print(f"Output : {out_dir}\n")

    for pose in poses:
        vid_ids = labels_df.loc[labels_df["class"] == pose, "vid_id"].astype(int).tolist()
        if not vid_ids:
            print(f"[SKIP] {pose}: tidak ada vid_id.")
            continue

        # ---------- LANDMARKS ----------
        lms_pose = landmarks_df[landmarks_df["vid_id"].astype(int).isin(vid_ids)].copy()
        lms_out_path = os.path.join(out_dir, f"{pose}_landmarks_raw.csv")
        lms_pose.to_csv(lms_out_path, index=False)
        print(f"[OK] Landmarks {pose} → {os.path.relpath(lms_out_path)}")

        # ---------- ANGLES ----------
        ang_pose = angles_df[angles_df["vid_id"].astype(int).isin(vid_ids)].copy()
        ang_out_path = os.path.join(out_dir, f"{pose}_angles_raw.csv")
        ang_pose.to_csv(ang_out_path, index=False)
        print(f"[OK] Angles {pose} → {os.path.relpath(ang_out_path)}")

    print("\nSelesai split data per pose (RAW, lengkap dengan vid_id & frame_order).")

if __name__ == "__main__":
    main(DEFAULT_DATA_DIR, DEFAULT_OUT_DIR)
