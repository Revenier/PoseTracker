#from app import data_loader as dl
from app import data_loader_csv as dl
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import redis, json, numpy as np
from app.redis_client import load_feature_matrix, r
import time

BACKEND = "redis"  # "csv" or "redis"
TOTAL_LOAD_TIME = 0.0   

def get_ref_landmarks(posture):
    if BACKEND == "redis":
        return get_ref_from_redis(posture, "landmarks")
    else:
        global TOTAL_LOAD_TIME
        t0 = time.time()
        ref = dl.posture_map[posture]["landmarks"]()
        load_time = time.time() - t0
        TOTAL_LOAD_TIME += load_time
        print(f"[landmark_logic][{BACKEND.upper()}] Loaded {posture}:landmark in {load_time:.4f}s")
        return np.asarray(ref, dtype=float)


def get_ref_angles(posture):
    if BACKEND == "redis":
        return get_ref_from_redis(posture, "angles")
    else:
        global TOTAL_LOAD_TIME
        t0 = time.time()
        ref = dl.posture_map[posture]["angles"]()
        load_time = time.time() - t0
        TOTAL_LOAD_TIME += load_time
        print(f"[angle_logic][{BACKEND.upper()}] Loaded {posture}:angle in {load_time:.4f}s")
        arr = np.asarray(ref, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

# TODO: cek lagi functionnnya bener ga buat ambil semua data dr redis, tolong sesuaiin sama punya lu. 
def get_ref_from_redis(posture, data_type):
    global TOTAL_LOAD_TIME
    alias = {"landmarks": "landmark", "angles": "angle"}
    dt = alias.get(str(data_type).lower(), str(data_type).lower())

    # langsung ambil dari Redis
    t0 = time.time()
    feats = load_feature_matrix(posture, dt)
    load_time = time.time() - t0
    TOTAL_LOAD_TIME += load_time

    # validasi hasil
    if feats.size == 0:
        raise ValueError(f"No reference data in Redis for {posture}:{dt}")

    if feats.ndim != 2:
        feats = np.asarray(feats, dtype=float)

    print(f"[{dt}_logic][{BACKEND.upper()}] ✅ Loaded {feats.shape[0]} samples for {posture}:{dt} in {load_time:.4f}s")
    return feats

def landmark_logic(posture, input_landmarks):
    total_start = time.time()

    if len(input_landmarks) != 99:
        return {'status': 'error', 'message': f'Input data must have 99 values (got {len(input_landmarks)})'}

    input_norm = normalize([input_landmarks], axis=1)
    
    ref_landmarks = get_ref_landmarks(posture)

    ref_norm = normalize(ref_landmarks, axis=1)
    if input_norm.shape[1] != ref_norm.shape[1]:
        return f"Input and reference dimensions do not match: {input_norm.shape[1]} vs {ref_norm.shape[1]}"
    sims = cosine_similarity(input_norm, ref_norm)[0]
    best_score = np.max(sims)
    best_idx = np.argmax(sims)

    print("\nDEBUG POSE:")
    print(f"Best similarity score: {best_score:.4f} at index {best_idx}")
    
    VERY_GOOD = 0.9
    GOOD = 0.8
    POOR = 0.7

    if best_score > VERY_GOOD:
        result = "Correct form!"
    else:
        input_pose = input_norm[0].reshape(33, 3)
        ref_pose = ref_norm[best_idx].reshape(33, 3)

        body_parts = {
            'arms': ([11,13,15,12,14,16], "arm position"),
            'legs': ([23,25,27,24,26,28], "leg position"),
            'torso': ([11,12,23,24], "body alignment"),
            'shoulders': ([11,12], "shoulder level"),
            'hips': ([23,24], "hip position")
        }
        
        issues = []
        for body_part, (indices, name) in body_parts.items():
            part_diff = np.mean([np.linalg.norm(input_pose[i] - ref_pose[i]) for i in indices])
            if part_diff > 0.1: 
                issues.append(name)
        
        if issues:
            if best_score > GOOD:
                feedback = f"Almost there! Check your {' and '.join(issues[:2])}"
            elif best_score > POOR:
                feedback = f"Form needs work. Focus on {' and '.join(issues[:2])}"
            else:
                feedback = f"Incorrect form. Major issues with {' and '.join(issues[:2])}"
        else:
            feedback = "Wrong form, try again!"
            
        result = f"{feedback}"

    total_time = time.time() - total_start
    print(f"[landmark_logic][{BACKEND.upper()}] 🕒 Total processing time: {total_time:.4f}s")
    print(f"[GLOBAL][{BACKEND.upper()}] ⏱ CUMULATIVE LOAD TIME: {TOTAL_LOAD_TIME:.4f}s")

    return result

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
    total_start = time.time()

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

    # TODO: (DONE) (NEED CHECK)  ref_angles should be load from redis cache

    ref_angles = get_ref_angles(posture)

    # Find the closest reference frame (smallest total angle difference)
    diffs_all = np.abs(ref_angles - input_angles)
    sum_diffs = np.sum(diffs_all, axis=1)
    best_idx = np.argmin(sum_diffs)
    best_ref = ref_angles[best_idx]
    diffs = np.abs(input_angles - best_ref)
    wrong_indices = np.where(diffs > 15)[0]  # threshold for "wrong"

    if len(wrong_indices) >= 3:
        result = f"You're not doing a {posture.replace('_', ' ')}. Please check your form."
    elif len(wrong_indices) > 0:
        max_idx = wrong_indices[np.argmax(diffs[wrong_indices])]
        suggestion = f"Try to adjust your {angle_names[max_idx]}: expected around {best_ref[max_idx]:.0f}°, got {input_angles[max_idx]:.0f}°."
        result = f"Incorrect posture, try again! {suggestion}"
    else:
        result = "Correct posture!"
    
    total_time = time.time() - total_start
    print(f"[angle_logic][{BACKEND.upper()}] 🕒 Total processing time: {total_time:.4f}s")
    print(f"[GLOBAL][{BACKEND.upper()}] ⏱ CUMULATIVE LOAD TIME: {TOTAL_LOAD_TIME:.4f}s")

    return result