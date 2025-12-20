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

def body_part_index_priority(posture):
    posture_priorities = {
        'push_up': {
            'indices': list(range(99)),  # whole body
            'description': 'shoulders, elbows, wrists, hips, knees'
        },
        'situp': {
            'indices': [0, 1, 2, 33, 34, 35, 36, 37, 38, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86],  # nose, shoulders, hips, knees, ankles
            'description': 'nose, shoulders, hips, knees, ankles'
        },
        'squat': {
            'indices': [0, 1, 2, 33, 34, 35, 36, 37, 38, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84],
            'description': 'nose, shoulders, hips, knees, ankles'
        },
        'jumping_jack': {
            'indices': list(range(99)), # whole body
            'description': 'whole body'
        }
    }
    
    return posture_priorities.get(posture, {
        'indices': list(range(99)),
        'description': 'all landmarks'
    })

# new logic landmark

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

    sims_temp = ref_norm.dot(q_norm) 
    best_idx_temp = int(np.argmax(sims_temp))
    ref_pose_temp = ref_norm[best_idx_temp]

    priority_config = body_part_index_priority(posture)
    priority_indices = priority_config['indices']

    # Normalize q_priority to unit vector
    q_norm[priority_indices] = ref_pose_temp[priority_indices]
    # Similarity using only priority points
    sims = ref_pose_temp.dot(q_norm)
    best_score = round(float(sims), 4)    
    
    thresholds = {
        'push_up':      {'VERY_GOOD': 0.99,  'GOOD': 0.96,  'POOR': 0.93},
        'squat':        {'VERY_GOOD': 0.96, 'GOOD': 0.93,  'POOR': 0.90},
        'situp':        {'VERY_GOOD': 0.98,  'GOOD': 0.95, 'POOR': 0.91},
        'jumping_jack': {'VERY_GOOD': 0.99,  'GOOD': 0.96,  'POOR': 0.93},
    }
    t = thresholds.get(posture, {'VERY_GOOD': 0.99, 'GOOD': 0.95, 'POOR': 0.93})
    VERY_GOOD = t['VERY_GOOD']
    GOOD = t['GOOD']
    POOR = t['POOR']

    input_pose = normalize([input_landmarks.flatten()], axis=1)[0].reshape(33, 3)
    ref_pose = ref_pose_temp.reshape(33, 3)

    # Body part definitions
    body_parts = {
        'arms': ([11,13,15,12,14,16], "arms"),
        'legs': ([23,25,27,24,26,28], "legs"),
        'torso': ([11,12,23,24], "torso"),
        'shoulders': ([11,12], "shoulders"),
        'hips': ([23,24], "hips")
    }
    
    # Calculate shoulder width as reference for relative measurements
    shoulder_width_input = np.linalg.norm(input_pose[11] - input_pose[12])
    shoulder_width_ref = np.linalg.norm(ref_pose[11] - ref_pose[12])
    
    issues = []
    detailed_feedback = []
    
    for part_name, (indices, display_name) in body_parts.items():
        part_diff = np.mean([np.linalg.norm(input_pose[i] - ref_pose[i]) for i in indices])
        
        if part_diff > 0.01:
            issues.append(display_name)
            
            # Get detailed directional feedback for this body part
            directions = get_directional_feedback(
                posture,
                part_name, 
                indices, 
                input_pose, 
                ref_pose,
            )
            
            if directions:
                detailed_feedback.extend(directions)
    
    # Limit to top 2 most important issues
    detailed_feedback = detailed_feedback[:2]
    
    if best_score > VERY_GOOD:
        result = {
            "correct": True,
            "status": "Perfect form",
            "feedback": "Perfect form! Keep it up!",
            "score": best_score,
        }
    elif issues:
        # Generate feedback with specific directions
        if detailed_feedback:
            feedback_text = " | ".join(detailed_feedback)
        else:
            feedback_text = f"Adjust your {' and '.join(issues[:2])}"
        
        if best_score > GOOD:
            correct = True
            status = "Good form"
            feedback = f"{feedback_text}"
        elif best_score > POOR:
            correct = False
            status = "Bad form"
            feedback = f"{feedback_text}"
        else:
            correct = False
            status = "Poor form"
            feedback = f"{feedback_text}"
        
        result = {
            "correct": correct,
            "status": status,
            "feedback": feedback,
            "score": best_score,
            "suggestions": detailed_feedback  # Separate field for detailed tips
        }
    else:
        result = {
            "correct": False,
            "status": "Incorrect",
            "feedback": "Wrong form, try again!",
            "score": best_score,
        }

    return result

def get_directional_feedback(posture, part_name, indices, input_pose, ref_pose):
   
    feedback = []

    # Posture-specific priorities and thresholds
    posture_config = {
        'push_up': {
            'priority_parts': ['arms', 'torso', 'shoulders'],
            'tips': {
                'arms': "Keep arms straight and shoulder-width apart",
                'torso': "Keep body in straight line, no sagging hips",
                'shoulders': "Shoulders should be over wrists"
            }
        },
        'squat': {
            'priority_parts': ['legs', 'torso', 'hips'],
            'tips': {
                'legs': "Keep knees aligned with toes, chest up",
                'torso': "Keep back straight, chest forward",
                'hips': "Lower hips down, knees shouldn't go past toes"
            }
        },
        'situp': {
            'priority_parts': ['torso', 'arms', 'shoulders'],
            'tips': {
                'torso': "Keep back straight, engage core",
                'arms': "Arms should be across chest or behind head",
                'shoulders': "Shoulders back, avoid hunching"
            }
        },
        'jumping_jack': {
            'priority_parts': ['arms', 'legs', 'shoulders'],
            'tips': {
                'arms': "Raise arms to shoulder height, synchronized movement",
                'legs': "Jump with feet shoulder-width apart",
                'shoulders': "Arms should reach ear level"
            }
        }
    }
    
    # Get config for current posture, default if not found
    config = posture_config.get(posture, {
        'priority_parts': ['torso', 'arms', 'legs'],
        'tips': {
            'arms': "Adjust arm position",
            'legs': "Adjust leg position",
            'torso': "Keep torso aligned"
        }
    })
    
    # Only provide feedback if this part is a priority for this posture
    if part_name not in config['priority_parts']:
        return feedback
    
    # Get posture-specific tip
    tip = config['tips'].get(part_name, f"Adjust {part_name}")
    
    # Calculate direction-specific corrections
    if part_name == 'arms':
        mean_input = np.mean(input_pose[indices], axis=0)
        mean_ref = np.mean(ref_pose[indices], axis=0)
        diff = mean_input - mean_ref
        
        if abs(diff[0]) > 0.05:
            feedback.append(f"{tip}. Move {'right' if diff[0] > 0 else 'left'}")
        if abs(diff[1]) > 0.05:
            feedback.append(f"{tip}. Move {'down' if diff[1] > 0 else 'up'}")
            
    elif part_name == 'legs':
        mean_input = np.mean(input_pose[indices], axis=0)
        mean_ref = np.mean(ref_pose[indices], axis=0)
        diff = mean_input - mean_ref
        
        if abs(diff[0]) > 0.05:
            feedback.append(f"{tip}. Position {'inward' if diff[0] > 0 else 'outward'}")
        if abs(diff[1]) > 0.05:
            feedback.append(f"{tip}. {'Lower' if diff[1] > 0 else 'Raise'} legs")
            
    elif part_name == 'torso':
        shoulder_diff = abs(input_pose[11][0] - input_pose[12][0]) - abs(ref_pose[11][0] - ref_pose[12][0])
        if abs(shoulder_diff) > 0.05:
            feedback.append(f"{tip}. Align shoulders level")
        
        # Check torso alignment
        mean_y_input = (input_pose[11][1] + input_pose[12][1]) / 2
        mean_y_ref = (ref_pose[11][1] + ref_pose[12][1]) / 2
        if abs(mean_y_input - mean_y_ref) > 0.05:
            feedback.append(f"{tip}. {'Straighten' if mean_y_input > mean_y_ref else 'Bend'} back")
            
    elif part_name == 'shoulders':
        left_input = input_pose[11]
        right_input = input_pose[12]
        left_ref = ref_pose[11]
        right_ref = ref_pose[12]
        
        left_diff = np.linalg.norm(left_input - left_ref)
        right_diff = np.linalg.norm(right_input - right_ref)
        
        if left_diff > 0.05:
            feedback.append(f"{tip}. Adjust left shoulder")
        if right_diff > 0.05:
            feedback.append(f"{tip}. Adjust right shoulder")
            
    elif part_name == 'hips':
        mean_input = np.mean(input_pose[indices], axis=0)
        mean_ref = np.mean(ref_pose[indices], axis=0)
        diff = mean_input - mean_ref
        
        if abs(diff[1]) > 0.05:
            feedback.append(f"{tip}. {'Lower' if diff[1] > 0 else 'Raise'} hips")
    
    return feedback[:2] 
