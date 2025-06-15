import pickle
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error
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

def validate_pickle_file(pickle_path, expected_type="DataFrame"):
    """
    Validate that pickle file exists and can be loaded properly.
    
    Args:
        pickle_path: Path to pickle file
        expected_type: Expected type of pickled object
    
    Returns:
        tuple: (success: bool, data: object, error_message: str)
    """
    if not os.path.exists(pickle_path):
        return False, None, f"Pickle file not found: {pickle_path}"
    
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        if expected_type == "DataFrame" and not isinstance(data, pd.DataFrame):
            return False, None, f"Expected DataFrame, got {type(data)}"
        
        if isinstance(data, pd.DataFrame) and data.empty:
            return False, None, "DataFrame is empty"
        
        return True, data, ""
    
    except Exception as e:
        return False, None, f"Failed to load pickle: {str(e)}"

def pushup_landmarks(input_landmarks):
    """
    Compares input_landmarks (1D numpy array) to push-up reference using cosine similarity.
    Returns detailed feedback with score.
    
    Args:
        input_landmarks: numpy array of landmark coordinates
    
    Returns:
        dict: Contains feedback message and similarity score
    """
    try:
        # Load and validate reference data
        pickle_path = os.path.join(pickles_dir, 'push_up_landmarks.pkl')
        success, ref_df, error_msg = validate_pickle_file(pickle_path)
        
        if not success:
            return {
                "message": error_msg,
                "similarity_score": 0.0,
                "error": True
            }
        
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
        
        # Handle dimension mismatch
        if ref.shape[1] != input_landmarks.shape[1]:
            # Try to pad or truncate to match dimensions
            ref_dims = ref.shape[1]
            input_dims = input_landmarks.shape[1]
            
            if input_dims < ref_dims:
                # Pad input with zeros
                padding = np.zeros((input_landmarks.shape[0], ref_dims - input_dims))
                input_landmarks = np.hstack([input_landmarks, padding])
                # print(f"Padded input from {input_dims} to {ref_dims} dimensions")
            elif input_dims > ref_dims:
                # Truncate input
                input_landmarks = input_landmarks[:, :ref_dims]
                # print(f"Truncated input from {input_dims} to {ref_dims} dimensions")
        
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
            "total_references": len(ref),
            "input_dimensions": input_landmarks.shape[1],
            "reference_dimensions": ref.shape[1]
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
        # Load and validate reference data
        pickle_path = os.path.join(pickles_dir, 'push_up_angles.pkl')
        success, ref_df, error_msg = validate_pickle_file(pickle_path)
        
        if not success:
            return {
                "message": error_msg,
                "mae_score": float('inf'),
                "error": True
            }
        
        # print(f"Loaded angles reference data shape: {ref_df.shape}")
        # print(f"Angles reference columns: {list(ref_df.columns)}")
        
        # Remove metadata columns safely
        ref_clean = safe_drop_columns(ref_df, ['vid_id', 'frame_order'])
        ref = ref_clean.values
        
        # print(f"Angles reference data after cleaning: {ref.shape}")
        # print(f"Input angles shape: {input_angles.shape}")
        
        # Ensure input is numpy array
        input_angles = np.array(input_angles)
        
        # Handle dimension mismatch
        if ref.shape[1] != len(input_angles):
            ref_dims = ref.shape[1]
            input_dims = len(input_angles)
            
            if input_dims < ref_dims:
                # Pad input with average of existing angles
                avg_angle = np.mean(input_angles) if len(input_angles) > 0 else 90.0
                padding = np.full(ref_dims - input_dims, avg_angle)
                input_angles = np.concatenate([input_angles, padding])
                # print(f"Padded angles from {input_dims} to {ref_dims} dimensions")
            elif input_dims > ref_dims:
                # Truncate input
                input_angles = input_angles[:ref_dims]
                # print(f"Truncated angles from {input_dims} to {ref_dims} dimensions")
        
        # Calculate Mean Absolute Error for each reference pose
        mae_scores = [mean_absolute_error(reference_pose, input_angles) for reference_pose in ref]
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
        for i, diff in enumerate(angle_differences):
            if i < len(angle_names) and diff > 15:
                problematic_angles.append({
                    "angle": angle_names[i],
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
            "problematic_angles": problematic_angles,
            "input_dimensions": len(input_angles),
            "reference_dimensions": ref.shape[1]
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        # print(f"Error in pushup_angles: {error_details}")
        return {
            "message": f"Error in angle analysis: {str(e)}",
            "mae_score": float('inf'),
            "error": True,
            "debug_info": error_details
        }