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

# def landmark_logic(posture, input_landmarks, ref_landmarks=None):

#      # ensure ref_norm precomputed and cached for posture
#     if posture not in _REF_CACHE:
#         ref_norm = normalize(ref_landmarks, axis=1)
#         _REF_CACHE[posture] = ref_norm
#     else:
#         ref_norm = _REF_CACHE[posture]

#     # prepare query (flatten + normalize)
#     q = np.asarray(input_landmarks.flatten(), dtype=np.float32)
#     q_norm = q / (np.linalg.norm(q) or 1.0)

#     sims = ref_norm.dot(q_norm) 
#     best_idx = int(np.argmax(sims))
#     best_score = round(float(sims[best_idx]), 4)
    
#     VERY_GOOD = 0.99
#     GOOD = 0.96
#     POOR = 0.93

#     input_pose = normalize([input_landmarks.flatten()], axis=1)[0].reshape(33, 3)
#     ref_pose = ref_norm[best_idx].reshape(33, 3)

#     body_parts = {
#         'arms': ([11,13,15,12,14,16], "arm position"),
#         'legs': ([23,25,27,24,26,28], "leg position"),
#         'torso': ([11,12,23,24], "body alignment"),
#         'shoulders': ([11,12], "shoulder level"),
#         'hips': ([23,24], "hip position")
#     }
        
#     issues = []
    
#     for part_name, (indices, name) in body_parts.items():
#         part_diff = np.mean([np.linalg.norm(input_pose[i] - ref_pose[i]) for i in indices])
#         if part_diff > 0.01: 
#             issues.append(name)
        
#     if best_score > VERY_GOOD:
#         result = {
#             "correct": True,
#             "status": "Perfect form",
#             "feedback": "Perfect form! Keep it up!",
#             "score": best_score,
#         }

#     elif issues:
#         if best_score > GOOD:
#             correct = False
#             status = "Good form"
#             feedback = f"Check your {' and '.join(issues[:2])}"
#         elif best_score > POOR:
#             correct = False
#             status = "Bad form"
#             feedback = f"Improve your {' and '.join(issues[:2])}"
#         else:
#             correct = False
#             status = "Poor form"
#             feedback = f"Focus on form: {' and '.join(issues[:2])}"
        
#         result = {
#             "correct": correct,
#             "status": status,
#             "feedback": feedback,
#             "score": best_score,
#         }
#     else:
#         result = {
#             "correct": False,
#             "status": "Incorrect",
#             "feedback": "Wrong form, try again!",
#             "score": best_score,
#         }

#     return result

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

    sims = ref_norm.dot(q_norm) 
    best_idx = int(np.argmax(sims))
    best_score = round(float(sims[best_idx]), 4)
    
    VERY_GOOD = 0.99
    GOOD = 0.96
    POOR = 0.93

    input_pose = normalize([input_landmarks.flatten()], axis=1)[0].reshape(33, 3)
    ref_pose = ref_norm[best_idx].reshape(33, 3)

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
                shoulder_width_input,
                shoulder_width_ref
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
            correct = False
            status = "Good form"
            feedback = f"Almost there! {feedback_text}"
        elif best_score > POOR:
            correct = False
            status = "Bad form"
            feedback = f"Need work: {feedback_text}"
        else:
            correct = False
            status = "Poor form"
            feedback = f"Focus: {feedback_text}"
        
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


def get_directional_feedback(posture, part_name, indices, input_pose, ref_pose, shoulder_width_input, shoulder_width_ref):

    feedback = []
    
    # Landmark mapping
    landmarks = {
        11: "left_shoulder", 12: "right_shoulder",
        13: "left_elbow", 14: "right_elbow",
        15: "left_wrist", 16: "right_wrist",
        23: "left_hip", 24: "right_hip",
        25: "left_knee", 26: "right_knee",
        27: "left_ankle", 28: "right_ankle"
    }
    
    if posture != 'squat':
        if part_name == 'arms':
            # Check arm width (wrists)
            left_wrist_input = input_pose[15]
            right_wrist_input = input_pose[16]
            left_wrist_ref = ref_pose[15]
            right_wrist_ref = ref_pose[16]
            
            arm_width_input = np.linalg.norm(left_wrist_input - right_wrist_input)
            arm_width_ref = np.linalg.norm(left_wrist_ref - right_wrist_ref)
            
            # Normalize by shoulder width
            relative_arm_width_input = arm_width_input / (shoulder_width_input or 1.0)
            relative_arm_width_ref = arm_width_ref / (shoulder_width_ref or 1.0)
            
            width_diff = relative_arm_width_input - relative_arm_width_ref
            print(f"arm width diff: {width_diff}")
            
            if(posture == 'jumping_jack'):
                if abs(width_diff) > 0.3: # Threshold: 25% difference
                    if width_diff > 0:
                        feedback.append("Hands too wide, bring them closer together")
                    else:
                        feedback.append("Hands too narrow, spread them wider")
            else:
                if abs(width_diff) > 0.15:  # Threshold: 15% difference
                    if width_diff > 0:
                        feedback.append("Hands too wide, bring them closer together")
                    else:
                        feedback.append("Hands too narrow, spread them wider")

            # Check arm height (vertical position)
            avg_wrist_y_input = (left_wrist_input[1] + right_wrist_input[1]) / 2
            avg_wrist_y_ref = (left_wrist_ref[1] + right_wrist_ref[1]) / 2
            height_diff = avg_wrist_y_input - avg_wrist_y_ref
            
            if abs(height_diff) > 0.1:
                if height_diff > 0:
                    feedback.append("Lower your hands")
                else:
                    feedback.append("Raise your hands higher")
    
    elif part_name == 'legs':
        # Check leg stance width (ankles)
        left_ankle_input = input_pose[27]
        right_ankle_input = input_pose[28]
        left_ankle_ref = ref_pose[27]
        right_ankle_ref = ref_pose[28]
        
        leg_width_input = np.linalg.norm(left_ankle_input - right_ankle_input)
        leg_width_ref = np.linalg.norm(left_ankle_ref - right_ankle_ref)
        
        relative_leg_width_input = leg_width_input / (shoulder_width_input or 1.0)
        relative_leg_width_ref = leg_width_ref / (shoulder_width_ref or 1.0)
        
        width_diff = relative_leg_width_input - relative_leg_width_ref
        
        if(posture == 'jumping_jack'):
            if abs(width_diff) > 0.3:
                if width_diff > 0:
                    feedback.append("Legs too wide, bring them closer")
                else:
                    feedback.append("Legs too narrow, widen your stance")
        else:
            if abs(width_diff) > 0.2:
                if width_diff > 0:
                    feedback.append("Feet too wide, bring them closer")
                else:
                    feedback.append("Feet too narrow, widen your stance")
        
        # Check knee bend (average knee height)
        avg_knee_y_input = (input_pose[25][1] + input_pose[26][1]) / 2
        avg_knee_y_ref = (ref_pose[25][1] + ref_pose[26][1]) / 2
        bend_diff = avg_knee_y_input - avg_knee_y_ref
        
        if abs(bend_diff) > 0.1:
            if bend_diff > 0:
                feedback.append("Bend your knees more")
            else:
                feedback.append("Straighten your legs a bit")
    
    elif part_name == 'torso':
        # Check torso angle/alignment
        left_shoulder = input_pose[11]
        right_shoulder = input_pose[12]
        left_hip = input_pose[23]
        right_hip = input_pose[24]
        
        # Check if torso is tilted forward/backward
        shoulder_center_input = (left_shoulder + right_shoulder) / 2
        hip_center_input = (left_hip + right_hip) / 2
        
        shoulder_center_ref = (ref_pose[11] + ref_pose[12]) / 2
        hip_center_ref = (ref_pose[23] + ref_pose[24]) / 2
        
        # Z-axis (depth) difference
        torso_lean_input = shoulder_center_input[2] - hip_center_input[2]
        torso_lean_ref = shoulder_center_ref[2] - hip_center_ref[2]
        lean_diff = torso_lean_input - torso_lean_ref
        
        if abs(lean_diff) > 0.1:
            if lean_diff > 0:
                feedback.append("Lean forward slightly")
            else:
                feedback.append("Keep your torso more upright")
    
    elif part_name == 'shoulders':
        # Check if shoulders are level
        left_shoulder = input_pose[11]
        right_shoulder = input_pose[12]
        
        shoulder_tilt_input = left_shoulder[1] - right_shoulder[1]
        shoulder_tilt_ref = ref_pose[11][1] - ref_pose[12][1]
        tilt_diff = abs(shoulder_tilt_input) - abs(shoulder_tilt_ref)
        
        if abs(tilt_diff) > 0.05:
            if shoulder_tilt_input > 0:
                feedback.append("Level your shoulders (left side higher)")
            elif shoulder_tilt_input < 0:
                feedback.append("Level your shoulders (right side higher)")
    
    elif part_name == 'hips':
        # Check hip alignment
        left_hip = input_pose[23]
        right_hip = input_pose[24]
        
        hip_tilt_input = left_hip[1] - right_hip[1]
        hip_tilt_ref = ref_pose[23][1] - ref_pose[24][1]
        tilt_diff = abs(hip_tilt_input) - abs(hip_tilt_ref)
        
        if abs(tilt_diff) > 0.05:
            feedback.append("Keep your hips level")
    
    return feedback