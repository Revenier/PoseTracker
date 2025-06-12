import pickle
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
pickles_dir = os.path.join(base_dir, '../app/pickles')

def pushup_landmarks(input_landmarks):
    """
    Compares input_landmarks (1D numpy array) to push-up reference using cosine similarity.
    Returns feedback string.
    """
    with open(os.path.join(pickles_dir, 'push_up_landmarks.pkl'), 'rb') as f:
        ref_df = pickle.load(f)
    ref = ref_df.drop(['vid_id', 'frame_order'], axis=1, errors='ignore').values
    ref_norm = normalize(ref, axis=1)
    input_norm = normalize([input_landmarks], axis=1)
    sims = cosine_similarity(input_norm, ref_norm)[0]
    best_score = np.max(sims)
    if best_score > 0.95:
        return "Correct form!"
    else:
        return "Leg should be wider."  # Example feedback

def pushup_angles(input_angles):
    """
    Compares input_angles (1D numpy array) to push-up reference using MAE.
    Returns feedback string.
    """
    with open(os.path.join(pickles_dir, 'push_up_angles.pkl'), 'rb') as f:
        ref_df = pickle.load(f)
    ref = ref_df.drop(['vid_id', 'frame_order'], axis=1, errors='ignore').values
    mae = np.mean(np.abs(ref - input_angles), axis=1)
    best_mae = np.min(mae)
    if best_mae < 10:
        return "Angles look good!"
    else:
        return "Check your joint angles!"