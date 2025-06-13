from app import data_loader as dl
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


def landmark_logic(posture, input_landmarks):
    ref_landmarks = dl.posture_map[posture]['landmarks']
    input_norm = normalize([input_landmarks], axis=1)
    ref_norm = normalize(ref_landmarks, axis=1)
    if input_norm.shape[1] != ref_norm.shape[1]:
        return f"Input and reference dimensions do not match: {input_norm.shape[1]} vs {ref_norm.shape[1]}"
    sims = cosine_similarity(input_norm, ref_norm)[0]
    best_score = np.max(sims)
    if best_score > 0.95:
        return "Correct form!"
    else:
        return "wrong form, try again!"

def angle_logic(posture, input_angles):
    ref_angles = dl.posture_map[posture]['angles']
    mae = np.mean(np.abs(ref_angles - input_angles), axis=1)
    best_mae = np.min(mae)
    if best_mae < 10:
        return "Angles look good!"
    else:
        return "Wrong angles, try again!"
    
    