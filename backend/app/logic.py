#from app import data_loader as dl
#from app import data_loader_csv as dl
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import redis, json, numpy as np
from app.redis_client import load_feature_matrix, r

# variable untuk menyimpan data yang sudah di-load dari Redis
_REF_CACHE = {}

# TODO: cek lagi functionnnya bener ga buat ambil semua data dr redis, tolong sesuaiin sama punya lu. 
def get_ref_from_redis(posture, data_type):
    alias = {"landmarks": "landmark", "angles": "angle"}
    dt = alias.get(str(data_type).lower(), str(data_type).lower())

    key = (posture, dt)

    # Cek apakah posture + datatype ini udah pernah di-load sebelumnya
    if key in _REF_CACHE:
        feats = _REF_CACHE[key]
        print(f"♻️ Using cached reference for {posture}:{dt}")
    else:
        # Kalau belum ada, mulai ambil dari Redis
        pattern = f"{posture}:{dt}:*"
        cursor = 0
        all_keys = []
        while True:
            cursor, keys = r.scan(cursor=cursor, match=pattern, count=1000)
            for k in keys:
                # Pastikan hasil key berupa string (kadang bytes)
                if isinstance(k, bytes):
                    k = k.decode()
                # Tambahkan hanya key yang benar-benar cocok dengan prefix posture:type
                if k.startswith(f"{posture}:{dt}:"):
                    all_keys.append(k)
            # Kalau cursor == 0 berarti sudah selesai scan semua key
            if cursor == 0:
                break

        # Tampilkan berapa banyak key ditemukan
        print(f"📦 Found {len(all_keys)} keys for {posture}:{dt}")
        for k in all_keys[:10]:
            print(f"   • {k}")
        if len(all_keys) > 10:
            print(f"   ... and {len(all_keys) - 10} more ...")

        # Ambil isinya dari Redis
        feats = load_feature_matrix(posture, dt)
        _REF_CACHE[key] = feats

    # Kalau Redis gak punya data posture + data_type ini
    if feats.size == 0:
        msg = f"No reference data in Redis for posture='{posture}' type='{dt}'"
        print(f"❌ {msg}")
        raise ValueError(msg)

    if feats.ndim != 2:
        feats = np.asarray(feats, dtype=float)
        _REF_CACHE[key] = feats

    print(f"✅ Got {len(feats)} samples for posture='{posture}' type='{dt}' | shape={feats.shape}")
    return feats

def landmark_logic(posture, input_landmarks):
    # 1. Check if the posture is valid
    # if posture not in dl.posture_map:
    #     return {'status': 'error', 'message': 'Unknown posture'}

    # 2. Check if input_data has 99 values (33 points * 3 coords)
    if len(input_landmarks) != 99:
        return {'status': 'error', 'message': f'Input data must have 99 values (got {len(input_landmarks)})'}

    input_norm = normalize([input_landmarks], axis=1)
    # TODO: (DONE) (NEED CHECK) ref_landmarks_func should be load from redis cache
    # ref_landmarks_func = dl.posture_map[posture]['landmarks']
    # ref_landmarks_func = get_ref_from_redis(posture, 'landmarks')
    # 3. Get reference landmarks and normalize them
    #ref_landmarks = ref_landmarks_func()  # shape: (N, 99)

    ref_landmarks = get_ref_from_redis(posture, 'landmarks')


    ref_norm = normalize(ref_landmarks, axis=1)
    if input_norm.shape[1] != ref_norm.shape[1]:
        return f"Input and reference dimensions do not match: {input_norm.shape[1]} vs {ref_norm.shape[1]}"
    sims = cosine_similarity(input_norm, ref_norm)[0]
    best_score = np.max(sims)
    if best_score > 0.95:
        return "Correct form!"
    else:
        return "Wrong form, try again!"

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return 0.0  # or np.nan, or skip this angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def align_landmarks(landmarks):
    # landmarks: (33, 3) array
    left_shoulder = landmarks[11][:2]
    right_shoulder = landmarks[12][:2]
    center = (left_shoulder + right_shoulder) / 2

    # Vector from right to left shoulder
    shoulder_vec = left_shoulder - right_shoulder
    angle = np.arctan2(shoulder_vec[1], shoulder_vec[0])
    rotation = -angle  # rotate so shoulders are horizontal

    # Rotation matrix
    rot_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation),  np.cos(rotation)]
    ])

    # Center and rotate all (x, y)
    xy = landmarks[:, :2] - center
    xy_rot = xy @ rot_matrix.T
    aligned = np.hstack([xy_rot, landmarks[:, 2:3]])
    return aligned

def angle_logic(posture, input_data):
    # 1. Check if the posture is valid
    # if posture not in dl.posture_map:
    #     return {'status': 'error', 'message': 'Unknown posture'}

    # 2. Check if input_data has 99 values (33 points * 3 coords)
    if len(input_data) != 99:
        return {'status': 'error', 'message': f'Input data must have 99 values (got {len(input_data)})'}

    # 3. Group into (33, 3) array
    landmarks = np.array(input_data).reshape((33, 3))
    landmarks = align_landmarks(landmarks)

    # 4. Calculate angles for specific joints (example indices)
    angle_indices = [
        (14, 12, 24),  # right_elbow, right_shoulder, right_hip
        (13, 11, 23),  # left_elbow, left_shoulder, left_hip
        (26, 24, 25),  # right_knee, right_hip, left_knee
        (24, 26, 28),  # right_hip, right_knee, right_ankle
        (23, 25, 27),  # left_hip, left_knee, left_ankle
        (16, 14, 12),  # right_wrist, right_elbow, right_shoulder
        (15, 13, 11),  # left_wrist, left_elbow, left_shoulder
    ]

    angle_names = [
        "right elbow", "left elbow", "right knee", "right hip", "left hip", "right wrist", "left wrist"
    ]

    input_angles = []
    for a, b, c in angle_indices:
        input_angles.append(calculate_angle(landmarks[a], landmarks[b], landmarks[c]))
    input_angles = np.array(input_angles)

    # TODO: (DONE) (NEED CHECK)  ref_angles_func should be load from redis cache
    # ref_angles_func = dl.posture_map[posture]['angles']
    # ref_angles_func = get_ref_from_redis(posture, 'angles')
    # ref_angles = ref_angles_func()  # shape: (N, num_angles)

    ref_angles = get_ref_from_redis(posture, 'angles')

    # Find the closest reference frame (smallest total angle difference)
    diffs_all = np.abs(ref_angles - input_angles)
    sum_diffs = np.sum(diffs_all, axis=1)
    best_idx = np.argmin(sum_diffs)
    best_ref = ref_angles[best_idx]
    diffs = np.abs(input_angles - best_ref)
    wrong_indices = np.where(diffs > 15)[0]  # threshold for "wrong"

    if len(wrong_indices) >= 3:
        return f"You're not doing a {posture.replace('_', ' ')}. Please check your form."
    elif len(wrong_indices) > 0:
        max_idx = wrong_indices[np.argmax(diffs[wrong_indices])]
        suggestion = f"Try to adjust your {angle_names[max_idx]}: expected around {best_ref[max_idx]:.0f}°, got {input_angles[max_idx]:.0f}°."
        return f"Incorrect posture, try again! {suggestion}"
    else:
        return "Correct posture!"