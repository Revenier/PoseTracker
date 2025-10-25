from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from app.group import group_landmark_feedback, group_angle_feedback
from app.logic import landmark_logic, angle_logic, get_ref_landmarks, get_ref_angles
from app import data_loader as dl

app = Flask(__name__)
CORS(app)

@app.route('/pose', methods=['POST'])
def receive_pose():
    data = request.get_json()
    posture = data.get('posture')
    mediapipe = data.get('mediapipe', [])

    if not posture or not isinstance(mediapipe, list) or len(mediapipe) == 0:
        return ({
            "status": "error",
            "message": "Invalid input — need 'posture' and 'mediapipe' list."
        }), 400
    
    try:
        ref_landmarks_all = get_ref_landmarks(posture)  # np.array (N,99)
    except Exception as e:
        ref_landmarks_all = None
        print(f"[WARN] gagal load landmark ref: {e}")

    try:
        ref_angles_all = get_ref_angles(posture)  # np.array (N,7)
    except Exception as e:
        ref_angles_all = None
        print(f"[WARN] gagal load angle ref: {e}")

    
    results = []
    correct_count = 0

    # Loop semua sample mediapipe yang dikirim (setiap sample = satu frame postur)
    for idx, arr in enumerate(mediapipe, start=1):
        print(f"\n[Data {idx}] Processing posture={posture} | Input length={len(arr)}\n", flush=True)   
        try:
            angle_result = angle_logic(posture, arr, ref_angles=ref_angles_all)
            landmark_result = landmark_logic(posture, arr, ref_landmarks=ref_landmarks_all)
        except Exception as e:
            results.append({
                "index": idx,
                "status": "error",
                "message": str(e)
            })
            continue

        # tentuin apakah array ini dianggap benar
        feedback = ""
        is_correct = False

        if isinstance(landmark_result, dict):
            feedback = landmark_result.get("feedback", "")
            if "Correct" in feedback:
                is_correct = True

        elif isinstance(landmark_result, str):
            feedback = landmark_result
            if "Correct" in feedback:
                is_correct = True

        results.append({
            "index": idx,
            "angles": angle_result,
            "landmarks": landmark_result,
            "feedback": feedback,
            "correct": is_correct
        })

        if is_correct:
            correct_count += 1

    total = len(results)
    summary = {
        "total_inputs": total,
        "correct": correct_count,
        "incorrect": total - correct_count,
        "posture": posture,
        "summary_result": (
            "All Correct ✅" if correct_count == total else
            ("Some Incorrect ❌" if correct_count > 0 else "All Failed ⚠️")
        )
    }

    # grouped feedback
    # feedback_list_for_grouping = []
    # angle_wrong_indices = []
    # if isinstance(angle_result, dict) and 'wrong_indices' in angle_result:
    #     angle_wrong_indices.extend(angle_result['wrong_indices'])
        
    # feedback_list_for_grouping.append({'wrong_indices': landmark_result.get('wrong_indices', []) if isinstance(landmark_result, dict) else []})
    # grouped_landmark_feedback = group_landmark_feedback(feedback_list_for_grouping)
    # grouped_angle_feedback = group_angle_feedback(angle_wrong_indices)

    # Combine feedback if both have issues not working yet
    # if "Correct" in grouped_landmark_feedback and "correct" in grouped_angle_feedback.lower():
    #     final_feedback = "All poses correct!"
    # else:
    #     feedback_parts = []
    #     if "Correct" not in grouped_landmark_feedback:
    #         feedback_parts.append(grouped_landmark_feedback)
    #     if "correct" not in grouped_angle_feedback.lower():
    #         feedback_parts.append(grouped_angle_feedback)
    #     final_feedback = " | ".join(feedback_parts)

    return ({
        "status": "success",
        "summary": summary,
        "details": results,
        "formated_feedback": "",
        "grouped_feedback": ""
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


