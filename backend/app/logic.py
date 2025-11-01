#from app import data_loader as dl
from app import data_loader_csv as dl
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import redis, json, numpy as np
from app.redis_client import load_feature_matrix, r

def get_ref_from_redis(posture, data_type):
    alias = {"landmarks": "landmark", "angles": "angle"}
    dt = alias.get(str(data_type).lower(), str(data_type).lower())

    feats = load_feature_matrix(posture, dt)
    # validasi hasil
    if feats.size == 0:
        raise ValueError(f"No reference data in Redis for {posture}:{dt}")

    if feats.ndim != 2:
        feats = np.asarray(feats, dtype=float)

    return feats

_REF_CACHE = {}
USE_FAISS = False

def landmark_logic(posture, input_landmarks, ref_landmarks=None):

     # ensure ref_norm precomputed and cached for posture
    if posture not in _REF_CACHE:
        ref_norm = normalize(ref_landmarks, axis=1)
        _REF_CACHE[posture] = ref_norm
    else:
        ref_norm = _REF_CACHE[posture]

    # prepare query (flatten + normalize)
    q = np.asarray(input_landmarks.flatten(), dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) or 1.0)

    sims = ref_norm.dot(q_norm) 
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    
    VERY_GOOD = 0.99
    GOOD = 0.96
    POOR = 0.93

    input_pose = normalize([input_landmarks.flatten()], axis=1)[0].reshape(33, 3)
    ref_pose = ref_norm[best_idx].reshape(33, 3)

    body_parts = {
        'arms': ([11,13,15,12,14,16], "arm position"),
        'legs': ([23,25,27,24,26,28], "leg position"),
        'torso': ([11,12,23,24], "body alignment"),
        'shoulders': ([11,12], "shoulder level"),
        'hips': ([23,24], "hip position")
    }
        
    issues = []
    difference = {} 
    
    for part_name, (indices, name) in body_parts.items():
        part_diff = np.mean([np.linalg.norm(input_pose[i] - ref_pose[i]) for i in indices])
        difference[part_name] = float(part_diff) 
        if part_diff > 0.01: 
            issues.append(name)
        
    if best_score > VERY_GOOD:
        result = {
            "correct": True,
            "status": "Very Good",
            "feedback": "Perfect form! Keep it up!",
            "score": best_score,
            "body_part_difference": difference
        }

    elif issues:
        if best_score > GOOD:
            correct = False
            status = "Good"
            feedback = f"Almost there! Check your {' and '.join(issues[:2])}"
        elif best_score > POOR:
            correct = False
            status = "Bad form"
            feedback = f"Need improvement: {' and '.join(issues[:2])}"
        else:
            correct = False
            status = "Poor form"
            feedback = f"Focus on form: {' and '.join(issues[:2])}"
        
        result = {
            "correct": correct,
            "status": status,
            "feedback": feedback,
            "score": best_score,
            "body_part_difference": difference
        }
    else:
        result = {
            "correct": False,
            "status": "Incorrect",
            "feedback": "Wrong form, try again!",
            "score": best_score,
            "body_part_difference": difference
        }

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

def angle_logic(posture, input_data, ref_angles=None):
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
        input_angles.append(calculate_angle(input_data[a], input_data[b], input_data[c]))
    input_angles = np.array(input_angles)

    # Find the closest reference frame (smallest total angle difference)
    diffs_all = np.abs(ref_angles - input_angles)
    sum_diffs = np.sum(diffs_all, axis=1)
    best_idx = np.argmin(sum_diffs)
    best_ref = ref_angles[best_idx]
    diffs = np.abs(input_angles - best_ref)
    wrong_indices = np.where(diffs > 15)[0]  # threshold for "wrong"

    if len(wrong_indices) >= 3:
        result = {
            "correct": False,
            "status": "major_issues",
            "feedback": f"You're not doing a {posture.replace('_', ' ')}. Please check your form.",
        }
    elif len(wrong_indices) > 0:
        max_idx = wrong_indices[np.argmax(diffs[wrong_indices])]
        suggestion = f"Try to adjust your {angle_names[max_idx]}: expected around {best_ref[max_idx]:.0f}°, got {input_angles[max_idx]:.0f}°."
        result = {
            "correct": False,
            "status": "minor_issues",
            "feedback": f"Incorrect posture, try again! {suggestion}",
        }
    else:
        result = {
            "correct": True,
            "status": "correct",
            "feedback": "Correct posture!",
        }
    
    return result