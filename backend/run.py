from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from app.group import group_landmark_feedback
from app.logic import landmark_logic, angle_logic, get_ref_from_redis, align_landmarks
import threading

REFERENCE_DATA = {}
def load_reference_data():
    global REFERENCE_DATA
    
    postures = ["push_up", "situp", "squat", "jumping_jack"]
    
    for posture in postures:
        try:
            landmarks = get_ref_from_redis(posture, "landmarks")
            angles = get_ref_from_redis(posture, "angles")
            REFERENCE_DATA[posture] = {
                "landmarks": landmarks,
                "angles": angles,
                "loaded_at": np.datetime64('now')
            }
            print(f"Loaded reference data for {posture}")
        except Exception as e:
            print(f"Failed to load reference data for {posture}: {e}")

def create_app():
    app = Flask(__name__)
    CORS(app)

    READY = {"done": False}

    def _bootstrap():
        try:
            print("Loading reference data (will preload if needed)...")
            # 1) isi Redis dulu (kalau kamu memang perlu)
            try:
                import preloaded_normalized_data as pre  # kalau mau preload Redis dari CSV
                pre.main()  # <-- kalau gak mau preload, baris ini bisa kamu komen
            except Exception as e:
                print(f"(skip preload or already done) {e}")
            # 2) isi REFERENCE_DATA global
            load_reference_data()
            print("Initialization complete!")
        finally:
            READY["done"] = True

    threading.Thread(target=_bootstrap, daemon=True).start()
    
    return app

app = create_app()

@app.route('/pose', methods=['POST'])
def receive_pose():
    data = request.get_json()
    posture = data.get('posture')
    mediapipe = data.get('mediapipe', [])

    if not posture or not isinstance(mediapipe, list) or len(mediapipe) == 0:
        return ({
            "status": False,
            "message": "Invalid input — need 'posture' and 'mediapipe' list."
        }), 400
    
    if posture not in REFERENCE_DATA:
        return ({
            "status": False,
            "message": f"Unknown posture: {posture}"
        }), 400

    ref_landmarks_all = REFERENCE_DATA[posture]["landmarks"]
    ref_angles_all = REFERENCE_DATA[posture]["angles"]
    
    
    results = []
    correct_count = 0

    # Loop semua sample mediapipe yang dikirim (setiap sample = satu frame postur)
    for idx, arr in enumerate(mediapipe, start=1):
        try:
            if not isinstance(arr, list):
                raise ValueError(f"Frame {idx}: Input must be a list, got {type(arr)}")
            
            if len(arr) != 99:
                raise ValueError(f"Frame {idx}: Expected 99 coordinates (33 landmarks × 3), got {len(arr)}")
            

            array = align_landmarks(np.array(arr).reshape((33, 3)))
            # angle_result = angle_logic(posture, array, ref_angles=ref_angles_all)
            landmark_result = landmark_logic(posture, array, ref_landmarks=ref_landmarks_all)

            results.append({
                "index": idx,
                "landmarks": landmark_result
            })
            
            if landmark_result.get('correct', False):
                correct_count += 1

        except Exception as e:
            print(f"Error processing frame {idx}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            # Add the traceback for debugging
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            
            results.append({
                "index": idx,
                "status": False,
                "message": str(e),
                "debug_info": {
                    "data_type": str(type(arr)),
                    "data_length": len(arr) if isinstance(arr, (list, np.ndarray)) else None,
                    "error_type": type(e).__name__
                }
        })

    total = len(results)
    # summary = {
    #     "total_inputs": total,
    #     "correct": correct_count,
    #     "incorrect": total - correct_count,
    #     "posture": posture,
    # }

    if(correct_count > (total-correct_count)):
        status = True
    else:
        status = False

    return ({
        "status": status,
        # "summary": summary,
        "fullFeedback": results,
        "formattedFeedback": group_landmark_feedback(results),
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)


