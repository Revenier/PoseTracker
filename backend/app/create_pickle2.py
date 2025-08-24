import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import MinMaxScaler, normalize

# Get absolute paths for data files
base_dir = os.path.dirname(os.path.abspath(__file__))
angles_path = os.path.abspath(os.path.join(base_dir, '../../data/angles.csv'))
labels_path = os.path.abspath(os.path.join(base_dir, '../../data/labels.csv'))
landmarks_path = os.path.abspath(os.path.join(base_dir, '../../data/landmarks.csv'))

# Output directory (absolute path)
output_dir = os.path.join(base_dir, 'AlignedPickles2')
os.makedirs(output_dir, exist_ok=True)

# Load data
angles = pd.read_csv(angles_path)
labels = pd.read_csv(labels_path)
landmarks = pd.read_csv(landmarks_path)

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return 0.0  # or np.nan, or skip this angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def align_pose_full(landmarks):
    # 1. Get key points
    head = landmarks[0][:2]

    # 2. Translate so head is at (0,0)
    xy = landmarks[:, :2] - head

    # 3. Compute centers
    shoulder_center = (xy[11] + xy[12]) / 2
    hip_center = (xy[23] + xy[24]) / 2

    # 4. Flip X if needed so left_shoulder is always to the left of right_shoulder
    if xy[11][0] > xy[12][0]:
        xy[:, 0] *= -1
        shoulder_center = (xy[11] + xy[12]) / 2
        hip_center = (xy[23] + xy[24]) / 2

    # 5. Rotate so the body axis (shoulder_center to hip_center) is vertical
    body_axis = hip_center - shoulder_center
    angle = np.arctan2(body_axis[0], body_axis[1])  # Note: swap x/y for vertical alignment
    rotation = -angle
    rot_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation),  np.cos(rotation)]
    ])
    xy_rot = xy @ rot_matrix.T

    # 6. Scale so shoulder distance is 1
    shoulder_dist = np.linalg.norm(xy_rot[11] - xy_rot[12])
    if shoulder_dist > 0:
        xy_rot = xy_rot / shoulder_dist

    # 7. Normalize Z (height) as well
    z = landmarks[:, 2:3]
    z = z - np.mean(z)
    z = z / (np.std(z) + 1e-8)

    aligned = np.hstack([xy_rot, z])
    return aligned

# def align_pose_strict(landmarks):
    head = landmarks[0][:2]
    xy = landmarks[:, :2] - head

    # Identify which shoulder is left/right in the original image
    orig_left = landmarks[11][:2]
    orig_right = landmarks[12][:2]
    if orig_left[0] > orig_right[0]:
        left_idx, right_idx = 11, 12
    else:
        left_idx, right_idx = 12, 11

    left_shoulder_xy = xy[left_idx]
    right_shoulder_xy = xy[right_idx]

    # Flip X if needed so left_shoulder is always to the left of right_shoulder
    if left_shoulder_xy[0] > right_shoulder_xy[0]:
        xy[:, 0] *= -1
        left_shoulder_xy = xy[left_idx]
        right_shoulder_xy = xy[right_idx]

    # Flip Y if head is below shoulders (optional, depends on your coordinate system)
    if (left_shoulder_xy[1] + right_shoulder_xy[1]) / 2 < 0:
        xy[:, 1] *= -1
        left_shoulder_xy = xy[left_idx]
        right_shoulder_xy = xy[right_idx]

    # Rotate so shoulders are horizontal
    shoulder_vec = left_shoulder_xy - right_shoulder_xy
    angle = np.arctan2(shoulder_vec[1], shoulder_vec[0])
    rotation = -angle
    rot_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation),  np.cos(rotation)]
    ])
    xy_rot = xy @ rot_matrix.T

    # Scale so shoulder distance is 1
    shoulder_dist = np.linalg.norm(left_shoulder_xy - right_shoulder_xy)
    if shoulder_dist > 0:
        xy_rot = xy_rot / shoulder_dist

    # Normalize Z (height) as well
    z = landmarks[:, 2:3]
    z = z - np.mean(z)
    z = z / (np.std(z) + 1e-8)

    aligned = np.hstack([xy_rot, z])
    return aligned

def align_head_shoulders(landmarks):
    # 1. Find head point (use nose, index 0)
    head = landmarks[0][:2]
    # 2. Find left and right shoulders
    left_shoulder = landmarks[11][:2]
    right_shoulder = landmarks[12][:2]

    # 3. Translate so head is at (0,0)
    xy = landmarks[:, :2] - head

    # 4. Compute shoulder vector after translation
    left_shoulder_xy = xy[11]
    right_shoulder_xy = xy[12]
    shoulder_vec = left_shoulder_xy - right_shoulder_xy

    # 5. Flip if needed so left_shoulder is always to the left of right_shoulder
    if left_shoulder_xy[0] < right_shoulder_xy[0]:
        xy[:, 0] *= -1
        left_shoulder_xy = xy[11]
        right_shoulder_xy = xy[12]
        shoulder_vec = left_shoulder_xy - right_shoulder_xy

    # 6. Rotate so shoulders are horizontal
    angle = np.arctan2(shoulder_vec[1], shoulder_vec[0])
    rotation = -angle
    rot_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation),  np.cos(rotation)]
    ])
    xy_rot = xy @ rot_matrix.T

    # 7. Scale so shoulder distance is 1
    shoulder_dist = np.linalg.norm(left_shoulder_xy - right_shoulder_xy)
    if shoulder_dist > 0:
        xy_rot = xy_rot / shoulder_dist

    # 8. Normalize Z (height) as well
    z = landmarks[:, 2:3]
    z = z - np.mean(z)
    z = z / (np.std(z) + 1e-8)

    aligned = np.hstack([xy_rot, z])
    return aligned

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

# Get unique postures
postures = labels['class'].unique()

angle_indices = [
    (14, 12, 24),  # right_elbow, right_shoulder, right_hip
    (13, 11, 23),  # left_elbow, left_shoulder, left_hip
    (26, 24, 25),  # right_knee, right_hip, left_knee
    (24, 26, 28),  # right_hip, right_knee, right_ankle
    (23, 25, 27),  # left_hip, left_knee, left_ankle
    (16, 14, 12),  # right_wrist, right_elbow, right_shoulder
    (15, 13, 11),  # left_wrist, left_elbow, left_shoulder
]

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
    
    # Normalize angles (feature-wise)
    scaler = MinMaxScaler()
    angles_norm = scaler.fit_transform(angles_only)

    # Normalize landmarks (row-wise)
    landmarks_norm = normalize(landmarks_only, axis=1)

    # Align all landmarks so they face the same direction
    aligned_landmarks = []
    for row in landmarks_only:
        lm = np.array(row).reshape((33, 3))
        aligned = align_pose_full(lm)  # This function checks and aligns each pose
        aligned_landmarks.append(aligned.flatten())
    aligned_landmarks = np.array(aligned_landmarks)

    # Save as pickle (just the numpy arrays)
    with open(os.path.join(output_dir, f'{posture}_landmarks_aligned.pkl'), 'wb') as f:
        pickle.dump(aligned_landmarks, f)

        
        
# Example
# 
# angles = np.array([
#     [30, 60, 90],
#     [45, 75, 105],
#     [60, 90, 120]
# ])
# scaler = MinMaxScaler()
# angles_norm = scaler.fit_transform(angles)
# print(angles_norm)
# Output:
# [[0.  0.  0. ]
#  [0.5 0.5 0.5]
#  [1.  1.  1. ]]

# landmarks = np.array([
#     [1, 2, 2],
#     [3, 0, 4]
# ])
# landmarks_norm = normalize(landmarks, axis=1)
# print(landmarks_norm)
# Output:
# [[0.333 0.667 0.667]
#  [0.6   0.    0.8  ]]