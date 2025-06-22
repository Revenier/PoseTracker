from app import data_loader as dl
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

def landmark_logic(posture, input_landmarks):
    # 1. Check if the posture is valid
    if posture not in dl.posture_map:
        return {'status': 'error', 'message': 'Unknown posture'}

    # 2. Check if input_data has 99 values (33 points * 3 coords)
    if len(input_landmarks) != 99:
        return {'status': 'error', 'message': f'Input data must have 99 values (got {len(input_landmarks)})'}

    input_norm = normalize([input_landmarks], axis=1)
    ref_landmarks_func = dl.posture_map[posture]['landmarks']
    ref_landmarks = ref_landmarks_func()  # shape: (N, 99)
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
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def angle_logic(posture, input_data):
    # 1. Check if the posture is valid
    if posture not in dl.posture_map:
        return {'status': 'error', 'message': 'Unknown posture'}

    # 2. Check if input_data has 99 values (33 points * 3 coords)
    if len(input_data) != 99:
        return {'status': 'error', 'message': f'Input data must have 99 values (got {len(input_data)})'}

    # 3. Group into (33, 3) array
    landmarks = np.array(input_data).reshape((33, 3))

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
    input_angles = []
    for a, b, c in angle_indices:
        input_angles.append(calculate_angle(landmarks[a], landmarks[b], landmarks[c]))
    input_angles = np.array(input_angles)

    # 5. Normalize data
    input_norm = normalize([input_angles], axis=1)

    # 6. Cosine similarity with reference angles
    ref_angles_func = dl.posture_map[posture]['angles']
    ref_angles = ref_angles_func()  # shape: (N, num_angles)
    ref_norm = normalize(ref_angles, axis=1)
    if input_norm.shape[1] != ref_norm.shape[1]:
        return {'status': 'error', 'message': f'Input and reference angle dimensions do not match: {input_norm.shape[1]} vs {ref_norm.shape[1]}'}
    sims = cosine_similarity(input_norm, ref_norm)[0]
    best_score = np.max(sims)

    # 7. Return result
    if best_score > 0.95:
        return "Correct posture!"
    else:
        return "Incorrect posture, try again!"