#from app import data_loader as dl
from app import data_loader_csv as dl
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import redis, json, numpy as np
from app.redis_client import load_feature_matrix, r

from collections import defaultdict
import re

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


def body_part_index_priority(posture):
    posture_priorities = {
        'push_up': {
            'indices': list(range(99)),  # whole body
            'description': 'shoulders, elbows, wrists, hips, knees'
        },
        'situp': {
            'indices': [0, 1, 2, 33, 34, 35, 36, 37, 38, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86],
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

def body_part_feedback(posture):
    body_parts = {}
    if posture == 'push_up':
        body_parts = {
        # 'shoulder_left': ([11], "left shoulder"),
        # 'shoulder_right': ([12], "right shoulder"),
        'elbow_left': ([13], "left elbow"),
        'elbow_right': ([14], "right elbow"),
        'wrist_left': ([15], "left wrist"),
        'wrist_right': ([16], "right wrist"),
        'hip': ([23, 24], "hip"),
        'knee_left': ([25], "left knee"),
        'knee_right': ([26], "right knee"),
        'ankle_left': ([27], "left ankle"),
        'ankle_right': ([28], "right ankle"),
    }
    elif posture == 'squat':
        body_parts = {
        'hip': ([23, 24], "hip"),
        'knee_left': ([25], "left knee"),
        'knee_right': ([26], "right knee"),
        'ankle_left': ([27], "left ankle"),
        'ankle_right': ([28], "right ankle"),
    }
        
    elif posture == 'situp':
        body_parts = {
        'head': ([0], "head"),
        'torso': ([11, 12, 23, 24], "torso"),
        'ankle_left': ([27], "left ankle"),
        'ankle_right': ([28], "right ankle"),
        # 'shoulder_left': ([11], "left shoulder"),
        # 'shoulder_right': ([12], "right shoulder"),
    }
        
    elif posture == 'jumping_jack':
        body_parts = {
        'arms_left': ([11, 13, 15], "left arm"),     
        'arms_right': ([12, 14, 16], "right arm"),    
        'legs_left': ([23, 25, 27], "left leg"),       
        'legs_right': ([24, 26, 28], "right leg"),     
        # 'shoulder_left': ([11], "left shoulder"),
        # 'shoulder_right': ([12], "right shoulder"),
    }
        
    return body_parts

def landmark_logic(posture, input_landmarks, facingRight, ref_landmarks=None):
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
    # get the reference pose
    ref_pose_temp = ref_norm[best_idx_temp]

    # Get priority indices for the posture
    priority_config = body_part_index_priority(posture)
    priority_indices = priority_config['indices']

    # copy reference pose to a new variable
    mixed_pose = ref_pose_temp.copy()
    # overwrite priority indices with input pose
    mixed_pose[priority_indices] = q_norm[priority_indices]

    # bandingin reference pose dengan mixed pose
    sims = ref_pose_temp.dot(mixed_pose)
    best_score = round(float(sims), 4)
    
    thresholds = {
        'push_up':      {'VERY_GOOD': 0.99,  'GOOD': 0.96,  'POOR': 0.93},
        'squat':        {'VERY_GOOD': 0.99, 'GOOD': 0.96,  'POOR': 0.93},
        'situp':        {'VERY_GOOD': 0.99,  'GOOD': 0.95, 'POOR': 0.93},
        'jumping_jack': {'VERY_GOOD': 0.99,  'GOOD': 0.96,  'POOR': 0.93},
    }
    t = thresholds.get(posture, {'VERY_GOOD': 0.99, 'GOOD': 0.95, 'POOR': 0.93})
    VERY_GOOD = t['VERY_GOOD']
    GOOD = t['GOOD']
    POOR = t['POOR']

    if best_score > VERY_GOOD:
       return {
            "correct": True,
            "issues": "Perfect form",
            "feedback": "Perfect form! Keep it up!",
            "score": best_score,
        }
    
    input_pose = mixed_pose.reshape(33, 3)
    ref_pose = ref_norm[best_idx_temp].reshape(33, 3)

    # Body part definitions
    body_parts = body_part_feedback(posture)
    issues = []
    for part_name, (indices, display_name) in body_parts.items():
        part_diff = np.mean([np.linalg.norm(input_pose[i] - ref_pose[i]) for i in indices])
        print(f"Debug: {part_name} difference: {part_diff}", flush=True)
        if part_diff > 0.005:
            issues.append((display_name, part_diff, part_name, indices))

    # Sort by part_diff descending and take top 2
    issues_sorted = sorted(issues, key=lambda x: x[1], reverse=True)[:2]

    detailed_feedback = []
    top_issue_names = []
    for display_name, part_diff, part_name, indices in issues_sorted:
        if part_diff > 0: 
            top_issue_names.append(display_name)
            directions = get_directional_feedback(
                posture,
                part_name,
                indices,
                input_pose,
                ref_pose,
                facingRight
            )
            # print(f"Debug: {part_name} directions: {directions}")
            if directions:
                detailed_feedback.extend(directions)

    grouped = defaultdict(list)
    for fb in detailed_feedback:
        m = re.match(r"Move (\w+) your (.+) slightly\.", fb)
        if m:
            direction, part = m.groups()
            grouped[direction].append(part)
        else:
            grouped[None].append(fb)

    result = []
    for direction, parts in grouped.items():
        if direction:
            if len(parts) > 1:
                result.append(f"Move {direction} your {' and '.join(parts)} slightly.")
            else:
                result.append(f"Move {direction} your {parts[0]} slightly.")
        else:
            result.extend(parts)

    final_feedback = " ".join(result)
    feedback_text =  final_feedback
    # feedback_text = " | ".join(detailed_feedback)
    issue_text = f"Adjust your {' and '.join(top_issue_names)}" if top_issue_names else ""

    if top_issue_names:
        if best_score > GOOD:
            correct = True
        elif best_score > POOR:
            correct = False
        else:
            correct = False
        result = {
            "correct": correct,
            "issues": issue_text,
            "feedback": feedback_text,
            "score": best_score,
        }
    else:
        result = {
            "correct": False,
            "issues": "Incorrect position",
            "feedback": "Wrong form, try again!",
            "score": best_score,
        }
    return result

def bak_directional_feedback(posture, part_name, indices, input_pose, ref_pose):
   
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

JOINT_THRESHOLDS = {

    # X: left/right, Y: up/down, Z: forward/backward (camera perspective)
    "squat": {
        "hip": {"x": 999, "y": 0.07, "z": 0.12},
        "knee_left": {"x": 0.02, "y": 0.06, "z": 0.10},
        "knee_right": {"x": 0.02, "y": 0.06, "z": 0.10},
        "ankle_left": {"x": 0.02, "y": 999, "z": 999},
        "ankle_right": {"x": 0.02, "y": 999, "z": 999},
    },
    # X: left/right, Y: up/down, Z: forward/backward (camera perspective)
    "push_up": {
        # "shoulder_left": {"x": 999, "y": 0.08, "z": 999},
        # "shoulder_right": {"x": 999, "y": 0.08, "z": 999},
        "elbow_left": {"x": 0.02, "y": 0.06, "z": 0.08},
        "elbow_right": {"x": 0.02, "y": 0.06, "z": 0.08},
        "wrist_left": {"x": 0.02, "y": 999, "z": 999},
        "wrist_right": {"x": 0.02, "y": 999, "z": 999},
        "hip": {"x": 999, "y": 0.05, "z": 0.08},
        "knee_left": {"x": 0.02, "y": 0.02, "z": 999},
        "knee_right": {"x": 0.02, "y": 0.02, "z": 999},
        "ankle_left": {"x": 0.02, "y": 999, "z": 999},
        "ankle_right": {"x": 0.02, "y": 999, "z": 999},
    },
    # X: left/right, Y: up/down, Z: forward/backward (camera perspective)
    "situp": {
        "head": {"x": 0.01, "y": 0.02, "z": 999},
        "torso": {"x": 0.08, "y": 0.03, "z": 0.06},
        "ankle_left": {"x": 0.06, "y": 0.02, "z": 999},
        "ankle_right": {"x": 0.06, "y": 0.02, "z": 999},
        # "shoulder_left": {"x": 999, "y": 0.02, "z": 0.05},
        # "shoulder_right": {"x": 999, "y": 0.02, "z": 0.05},
    },
    # X: left/right, Y: up/down, Z: forward/backward (camera perspective)
    "jumping_jack": {
        "arms_left": {"x": 0.05, "y": 0.05, "z": 999},
        "arms_right": {"x": 0.05, "y": 0.05, "z": 999},
        "legs_left": {"x": 0.06, "y": 0.06, "z": 999},
        "legs_right": {"x": 0.06, "y": 0.06, "z": 999},
        # "shoulder_left": {"x": 0.05, "y": 0.05, "z": 999},
        # "shoulder_right": {"x": 0.05, "y": 0.05, "z": 999},
    },
}

# AXIS FEEDBACK MAP dengan axis yang bisa di-skip (None = skip)
AXIS_FEEDBACK_MAP = {
    "squat": {
        "hip": {"x": "horizontal", "y": "lower", "z": "forward"},
        "knee_left": {"x": "align", "y": "bend", "z": "forward"},
        "knee_right": {"x": "align", "y": "bend", "z": "forward"},
        "ankle_left": {"x": "align", "y": None, "z": None},
        "ankle_right": {"x": "align", "y": None, "z": None},
    },
    "push_up": {
        # "shoulder_left": {"x": "align", "y": "level", "z": "forward"},
        # "shoulder_right": {"x": "align", "y": "level", "z": "forward"},
        "elbow_left": {"x": "align", "y": "bend", "z": "forward"},
        "elbow_right": {"x": "align", "y": "bend", "z": "forward"},
        "wrist_left": {"x": "align", "y": "level", "z": "forward"},
        "wrist_right": {"x": "align", "y": "level", "z": "forward"},
        "hip": {"x": "align", "y": "level", "z": "forward"},
        "knee_left": {"x": "align", "y": "straight", "z": None},
        "knee_right": {"x": "align", "y": "straight", "z": None},
        "ankle_left": {"x": "align", "y": None, "z": None},
        "ankle_right": {"x": "align", "y": None, "z": None},
    },
    "situp": {
        "head": {"x": "align", "y": "raise", "z": "forward"},
        "torso": {"x": "center", "y": "raise", "z": "forward"},
        "ankle_left": {"x": "align", "y": "raise", "z": None},
        "ankle_right": {"x": "align", "y": "raise", "z": None},
        # "shoulder_left": {"x": "align", "y": "raise", "z": "forward"},
        # "shoulder_right": {"x": "align", "y": "raise", "z": "forward"},
    },
    "jumping_jack": {
        "arms_left": {"x": "spread", "y": "raise", "z": "forward"},
        "arms_right": {"x": "spread", "y": "raise", "z": "forward"},
        "legs_left": {"x": "spread", "y": "jump", "z": "forward"},
        "legs_right": {"x": "spread", "y": "jump", "z": "forward"},
        # "shoulder_left": {"x": "align", "y": "raise", "z": "forward"},
        # "shoulder_right": {"x": "align", "y": "raise", "z": "forward"},
    },
}

def get_directional_feedback(posture, part_name, indices, input_pose, ref_pose, facingRight):
    """
    Generate actionable feedback for a specific body part by comparing input_pose and ref_pose.
    Uses static thresholds per joint per posture.
    """
    feedback = []
    body_parts = body_part_feedback(posture)
    display_name = body_parts.get(part_name, (indices, part_name))[1]

    mean_input = np.mean(input_pose[indices], axis=0)
    mean_ref = np.mean(ref_pose[indices], axis=0)
    diff = mean_input - mean_ref

    # Ambil threshold spesifik untuk joint ini
    posture_thresholds = JOINT_THRESHOLDS.get(posture, {})
    part_thresholds = posture_thresholds.get(part_name, {})
    
    x_threshold = part_thresholds.get("x")
    print(f"Debug: {part_name} x diff: {diff[0]}, threshold: {x_threshold}", flush=True)
    y_threshold = part_thresholds.get("y")
    z_threshold = part_thresholds.get("z")
    
    # Ambil feedback map
    posture_map = AXIS_FEEDBACK_MAP.get(posture, {})
    part_map = posture_map.get(part_name, {"x": "move", "y": "adjust", "z": "move"})

    directions = []
    
    # X-axis (left/right)
    if x_threshold is not None and abs(diff[0]) > x_threshold:
        x_feedback_type = part_map.get("x", "move")
        if x_feedback_type == "move":
            directions.append(f"move {('right' if diff[0] > 0 else 'left')}")
        elif x_feedback_type == "align":
            # print(f"debug here!! {facingRight} , diff: {diff[0]}", flush=True)
            if facingRight == True:
                directions.append(f"{'move forward' if diff[0] > 0 else 'move backward'}")
            elif facingRight == False:
                directions.append(f"{'move backward' if diff[0] > 0 else 'move forward'}")
        elif x_feedback_type == "spread":
            directions.append(f"{'spread wider' if diff[0] > 0 else 'bring closer'}")
        elif x_feedback_type == "center":
            directions.append(f"center your position")
    
    # Y-axis (up/down)
    if y_threshold is not None and abs(diff[1]) > y_threshold:
        y_feedback_type = part_map.get("y")
        if y_feedback_type is None:
            pass
        elif y_feedback_type == "raise":
            if posture == "situp":
                directions.append(f"{'lower'}")
            else:
                directions.append(f"{'raise' if diff[1] < 0 else 'lower'}")
        elif y_feedback_type == "lower":
            directions.append(f"{'lower' if diff[1] > 0 else 'raise'}")
        elif y_feedback_type == "bend":
            directions.append(f"{'bend more' if diff[1] > 0 else 'straighten'}")
        elif y_feedback_type == "straight":
            directions.append(f"keep legs straight")
        elif y_feedback_type == "level":
            directions.append(f"keep level")
        elif y_feedback_type == "jump":
            directions.append(f"{'jump higher' if diff[1] > 0 else 'jump lower'}")
    
    # Z-axis (forward/backward)
    if z_threshold is not None and abs(diff[2]) > z_threshold:
        z_feedback_type = part_map.get("z")
        if z_feedback_type is None:
            pass
        elif z_feedback_type == "forward":
            directions.append(f"move {'forward' if diff[2] > 0 else 'backward'}")
        elif z_feedback_type == "move":
            directions.append(f"move {'forward' if diff[2] > 0 else 'backward'}")

    # print(f"[Diff] {part_name}: x={diff[0]:.4f}, y={diff[1]:.4f}, z={diff[2]:.4f}", flush=True)
    # print(f"[Thresholds] {part_name}: x={x_threshold}, y={y_threshold}, z={z_threshold}", flush=True)

    if directions:
        # movement = " and ".join(directions)
        feedback.append(f"{directions[0].capitalize()} your {display_name} slightly.")
    else:
        feedback.append(f"Move your {display_name} position.")

    # print(f"[DIRECTION] part={display_name}, triggered={directions}", flush=True)

    return feedback[:2]