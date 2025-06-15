# import pickle
# import numpy as np
# from sklearn.preprocessing import normalize
# from sklearn.metrics.pairwise import cosine_similarity
# import os

# base_dir = os.path.dirname(os.path.abspath(__file__))
# pickles_dir = os.path.join(base_dir, '../app/pickles')

# def pushup_landmarks(input_landmarks):
#     """
#     Compares input_landmarks (1D numpy array) to push-up reference using cosine similarity.
#     Returns feedback string.
#     """
#     with open(os.path.join(pickles_dir, 'push_up_landmarks.pkl'), 'rb') as f:
#         ref_df = pickle.load(f)
#     ref = ref_df.drop(['vid_id', 'frame_order'], axis=1, errors='ignore').values
#     ref_norm = normalize(ref, axis=1)
#     input_norm = normalize([input_landmarks], axis=1)
#     sims = cosine_similarity(input_norm, ref_norm)[0]
#     best_score = np.max(sims)
#     if best_score > 0.95:
#         return "Correct form!"
#     else:
#         return "Leg should be wider."  # Example feedback

# def pushup_angles(input_angles):
#     """
#     Compares input_angles (1D numpy array) to push-up reference using MAE.
#     Returns feedback string.
#     """
#     with open(os.path.join(pickles_dir, 'push_up_angles.pkl'), 'rb') as f:
#         ref_df = pickle.load(f)
#     ref = ref_df.drop(['vid_id', 'frame_order'], axis=1, errors='ignore').values
#     mae = np.mean(np.abs(ref - input_angles), axis=1)
#     best_mae = np.min(mae)
#     if best_mae < 10:
#         return "Angles look good!"
#     else:
#         return "Check your joint angles!"

import pickle
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
pickles_dir = os.path.join(base_dir, 'pickles')

def safe_drop_columns(df, columns_to_drop):
    """
    Safely remove columns from DataFrame, handling pandas version compatibility issues.
    """
    try:
        # First approach - standard drop
        return df.drop(columns_to_drop, axis=1, errors='ignore')
    except Exception:
        try:
            # Second approach - select columns that are NOT in the drop list
            cols_to_keep = [col for col in df.columns if col not in columns_to_drop]
            return df[cols_to_keep]
        except Exception:
            # Third approach - select only numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            return df[numeric_cols]

def pushup_landmarks(input_landmarks):
    """
    Compares input_landmarks (1D numpy array) to push-up reference using cosine similarity.
    Returns detailed feedback with score.
    
    Args:
        input_landmarks: numpy array of 99 landmark coordinates (33 points * 3 coords)
    
    Returns:
        dict: Contains feedback message and similarity score
    """
    try:
        # Load reference data
        pickle_path = os.path.join(pickles_dir, 'push_up_landmarks.pkl')
        
        if not os.path.exists(pickle_path):
            return {
                "message": f"Reference file not found: {pickle_path}",
                "similarity_score": 0.0,
                "error": True
            }
        
        with open(pickle_path, 'rb') as f:
            ref_df = pickle.load(f)
        
        # print(f"Loaded reference data shape: {ref_df.shape}")
        # print(f"Reference columns: {list(ref_df.columns)}")
        
        # Remove metadata columns safely
        ref_clean = safe_drop_columns(ref_df, ['vid_id', 'frame_order'])
        ref = ref_clean.values
        
        # print(f"Reference data after cleaning: {ref.shape}")
        # print(f"Input landmarks shape: {input_landmarks.shape}")
        
        # Ensure input is the right shape
        if len(input_landmarks.shape) == 1:
            input_landmarks = input_landmarks.reshape(1, -1)
        
        # Check if dimensions match
        if ref.shape[1] != input_landmarks.shape[1]:
            return {
                "message": f"Dimension mismatch: reference has {ref.shape[1]} features, input has {input_landmarks.shape[1]}",
                "similarity_score": 0.0,
                "error": True
            }
        
        # Normalize both reference and input data
        ref_norm = normalize(ref, axis=1)
        input_norm = normalize(input_landmarks, axis=1)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(input_norm, ref_norm)[0]
        best_score = np.max(similarities)
        best_match_idx = np.argmax(similarities)
        
        # Generate detailed feedback based on similarity score
        if best_score > 0.95:
            feedback = "Excellent form! Your pose matches the reference perfectly."
        elif best_score > 0.90:
            feedback = "Good form! Minor adjustments might help."
        elif best_score > 0.80:
            feedback = "Fair form. Consider checking your body alignment."
        elif best_score > 0.70:
            feedback = "Form needs improvement. Focus on proper positioning."
        else:
            feedback = "Poor form detected. Please review proper push-up technique."
        
        return {
            "message": feedback,
            "similarity_score": float(best_score),
            "best_match_frame": int(best_match_idx),
            "total_references": len(ref)
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in pushup_landmarks: {error_details}")
        return {
            "message": f"Error in landmark analysis: {str(e)}",
            "similarity_score": 0.0,
            "error": True,
            "debug_info": error_details
        }

def pushup_angles(input_angles):
    """
    Compares input_angles (1D numpy array) to push-up reference using MAE.
    Returns detailed feedback with error metrics.
    
    Args:
        input_angles: numpy array of joint angles
    
    Returns:
        dict: Contains feedback message and error metrics
    """
    try:
        # Load reference data
        pickle_path = os.path.join(pickles_dir, 'push_up_angles.pkl')
        
        if not os.path.exists(pickle_path):
            return {
                "message": f"Reference file not found: {pickle_path}",
                "mae_score": float('inf'),
                "error": True
            }
        
            with open(pickle_path, 'rb') as f:
                ref_df = pickle.load(f)
        
        # print(f"Loaded angles reference data shape: {ref_df.shape}")
        # print(f"Angles reference columns: {list(ref_df.columns)}")
        
        # Remove metadata columns safely
        ref_clean = safe_drop_columns(ref_df, ['vid_id', 'frame_order'])
        ref = ref_clean.values
        
        # print(f"Angles reference data after cleaning: {ref.shape}")
        # print(f"Input angles shape: {input_angles.shape}")
        
        # Ensure input is numpy array
        input_angles = np.array(input_angles)
        
        # Check if dimensions match
        if ref.shape[1] != len(input_angles):
            return {
                "message": f"Angle dimension mismatch: reference has {ref.shape[1]} angles, input has {len(input_angles)}",
                "mae_score": float('inf'),
                "error": True
            }
        
        # Calculate Mean Absolute Error for each reference pose
        mae_scores = np.mean(np.abs(ref - input_angles), axis=1)
        best_mae = np.min(mae_scores)
        best_match_idx = np.argmin(mae_scores)
        
        # Generate detailed feedback based on MAE
        if best_mae < 5:
            feedback = "Perfect angles! Your joint positioning is excellent."
            rating = "Excellent"
        elif best_mae < 10:
            feedback = "Good angles! Minor tweaks to joint positioning recommended."
            rating = "Good"
        elif best_mae < 20:
            feedback = "Fair angles. Check your elbow and hip positioning."
            rating = "Fair"
        elif best_mae < 30:
            feedback = "Angles need improvement. Focus on proper joint alignment."
            rating = "Poor"
        else:
            feedback = "Significant angle issues detected. Review push-up form fundamentals."
            rating = "Very Poor"
        
        # Provide specific angle feedback
        angle_names = [
            "right_elbow_shoulder_hip", "left_elbow_shoulder_hip", 
            "knee_hip_alignment", "right_hip_knee_ankle", 
            "left_hip_knee_ankle", "right_wrist_elbow_shoulder", 
            "left_wrist_elbow_shoulder"
        ]
        
        # Find the reference pose with best match for detailed comparison
        best_ref = ref[best_match_idx]
        angle_differences = np.abs(best_ref - input_angles)
        
        # Identify problematic angles (difference > 15 degrees)
        problematic_angles = []
        for i, (angle_name, diff) in enumerate(zip(angle_names[:len(input_angles)], angle_differences)):
            if diff > 15:
                problematic_angles.append({
                    "angle": angle_name,
                    "your_angle": float(input_angles[i]),
                    "reference_angle": float(best_ref[i]),
                    "difference": float(diff)
                })
        
        return {
            "message": feedback,
            "rating": rating,
            "mae_score": float(best_mae),
            "best_match_frame": int(best_match_idx),
            "total_references": len(ref),
            "problematic_angles": problematic_angles
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in pushup_angles: {error_details}")
        return {
            "message": f"Error in angle analysis: {str(e)}",
            "mae_score": float('inf'),
            "error": True,
            "debug_info": error_details
        }

