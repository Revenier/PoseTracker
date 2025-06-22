import pandas as pd
import pickle
import os

# Get absolute paths for data files
base_dir = os.path.dirname(os.path.abspath(__file__))
angles_path = os.path.abspath(os.path.join(base_dir, '../../data/angles.csv'))
labels_path = os.path.abspath(os.path.join(base_dir, '../../data/labels.csv'))
landmarks_path = os.path.abspath(os.path.join(base_dir, '../../data/landmarks.csv'))

# Output directory (absolute path)
output_dir = os.path.join(base_dir, 'pickles')
os.makedirs(output_dir, exist_ok=True)

# Load data
angles = pd.read_csv(angles_path)
labels = pd.read_csv(labels_path)
landmarks = pd.read_csv(landmarks_path)

# Get unique postures
postures = labels['class'].unique()

for posture in postures:
    # Get vid_ids for this posture
    vids = labels[labels['class'] == posture]['vid_id'].astype(int).tolist()
    
    # Filter angles and landmarks for these vids
    angles_posture = angles[angles['vid_id'].astype(int).isin(vids)]
    landmarks_posture = landmarks[landmarks['vid_id'].astype(int).isin(vids)]
    
    # --- FIX: Only keep the raw values, drop id/frame columns ---
    # For angles, drop 'vid_id' and 'frame_order'
    angle_cols = [col for col in angles_posture.columns if col not in ['vid_id', 'frame_order']]
    angles_only = angles_posture[angle_cols].values

    # For landmarks, drop 'vid_id' and 'frame_order' if present
    landmark_cols = [col for col in landmarks_posture.columns if col not in ['vid_id', 'frame_order']]
    landmarks_only = landmarks_posture[landmark_cols].values

    # Save as pickle (just the numpy arrays)
    with open(os.path.join(output_dir, f'{posture}_angles.pkl'), 'wb') as f:
        pickle.dump(angles_only, f)
    with open(os.path.join(output_dir, f'{posture}_landmarks.pkl'), 'wb') as f:
        pickle.dump(landmarks_only, f)