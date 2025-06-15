# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import pandas as pd

# from app.pose_logic import pushup_landmarks, pushup_angles

# app = Flask(__name__)
# CORS(app)

# # Test data from your CSV samples (hardcoded for testing)
# SAMPLE_TEST_DATA = {
#     "landmarks": [
#         0,208,-0.14172298,-55.86816,-83.562836,0.77226424,-57.81948,-78.75882,1.233074,-57.699406
#         ,-78.759254,1.7493129,-57.53372,-78.78233,-1.3052675,-57.72683,-79.61645,-2.2151628,-57.54617
#         ,-79.630646,-3.1707785,-57.298553,-79.6179,2.756506,-55.85471,-50.886845,-4.7613044,-55.44263
#         ,-54.861935,1.1918291,-53.003086,-73.335304,-1.6319023,-52.80554,-74.4462,9.726373,-40.375786
#         ,-32.200623,-10.973151,-39.614494,-36.86989,26.051504,-39.137066,-31.208908,-27.616627,-37.245445
#         ,-37.595387,38.416695,-47.548244,-57.050426,-41.644993,-45.20416,-64.12199,42.235176,-49.71335
#         ,-63.664894,-45.83316,-47.217133,-71.61033,42.45105,-51.35462,-70.81914,-45.726147,-48.86608,-78.51114
#         ,40.822147,-50.241287,-61.593933,-44.12522,-47.887577,-68.40401,5.9720097,-0.1093643,0.26248583
#         ,-5.9720097,0.1093643,-0.26248583,11.75684,31.83908,-4.643736,-11.081656,32.370235,-5.003058,14.448328
#         ,61.644,31.237148,-14.260321,62.669857,27.827307,13.20743,64.77579,32.967335
#         ,-13.362365,66.36444,29.356094,16.494827,74.145805,1.6187178,-16.030586,74.7956,-1.0190959
#     ],
#     "angles": [0,208,90.98465,88.599945,38.76677,124.538,121.06854,116.595764,109.79157]
# }

# SAMPLE_TEST_DATA_2 = {
#     "landmarks": [0,214,-0.09433113,-52.82708,-57.42315,0.5958145,-54.782166,-51.497883,1.2533442
#     ,-54.74263,-51.483875,1.7506169,-54.66264,-51.489212,-1.3231053,-54.442043,-51.81584,-2.0938802
#     ,-54.147053,-51.82886,-2.8000605,-53.831818,-51.82372,2.6459832,-52.775394,-25.84235,-3.6822598
#     ,-51.875507,-26.83165,1.3602847,-49.7299,-48.547035,-1.1088375,-49.32755,-48.91358,8.897671,-39.977417
#     ,-19.134594,-8.778292,-40.022503,-19.594484,17.166763,-56.413048,-35.11141,-18.693428,-55.127407,-35.151936
#     ,12.452961,-73.303955,-49.73784,-14.891015,-74.892845,-45.190125,11.785952,-78.445435,-57.70145,-13.212663
#     ,-80.25727,-54.24714,10.43839,-78.664055,-56.0308,-11.621859,-80.60166,-50.147144,10.34366,-76.88346,-50.40773
#     ,-11.851206,-78.92865,-44.997295,5.7404513,-0.35265207,0.33231744,-5.7404513,0.35264072,-0.33231744,14.389475,27.73498
#     ,-0.55136734,-10.437229,28.616667,0.030994494,19.635443,51.882656,45.14488,-19.634829,51.199955,43.19759,18.795805
#     ,55.034233,48.25898,-20.250454,54.455738,46.124863,21.455263,60.002815,19.192198,-21.453554,59.534573,18.457273],
#     "angles": [0,214,151.51472,156.4084,47.469006,119.39075,116.57121,141.04236,148.08401]
# }

# # Map posture names to comparison functions
# COMPARISON_FUNCTIONS = {
#     "push_up": {
#         "landmarks": pushup_landmarks,
#         "angles": pushup_angles
#     }
# }

# @app.route('/pose', methods=['POST'])
# def receive_pose():
#     """
#     Main endpoint for pose analysis
#     Expected JSON format:
#     {
#         "posture": "push_up",
#         "landmarks": [99 float values],
#         "angles": [7 float values]
#     }
#     """
#     try:
#         data = request.get_json()
#         posture = data.get('posture')
#         landmarks = data.get('landmarks', [])
#         angles = data.get('angles', [])

#         if posture not in COMPARISON_FUNCTIONS:
#             return jsonify({
#                 'status': 'error', 
#                 'message': f'Unknown posture: {posture}. Supported: {list(COMPARISON_FUNCTIONS.keys())}'
#             }), 400

#         feedback = {}
        
#         # Process landmarks if provided
#         if landmarks:
#             try:
#                 landmarks_array = np.array(landmarks, dtype=float)
#                 if len(landmarks_array) != 99:
#                     return jsonify({
#                         'status': 'error',
#                         'message': f'Expected 99 landmark values, got {len(landmarks_array)}'
#                     }), 400
                
#                 landmarks_feedback = COMPARISON_FUNCTIONS[posture]["landmarks"](landmarks_array)
#                 feedback['landmarks'] = landmarks_feedback
#             except Exception as e:
#                 feedback['landmarks'] = f"Error processing landmarks: {str(e)}"

#         # Process angles if provided
#         if angles:
#             try:
#                 angles_array = np.array(angles, dtype=float)
#                 angles_feedback = COMPARISON_FUNCTIONS[posture]["angles"](angles_array)
#                 feedback['angles'] = angles_feedback
#             except Exception as e:
#                 feedback['angles'] = f"Error processing angles: {str(e)}"

#         return jsonify({
#             'status': 'success',
#             'posture': posture,
#             'feedback': feedback
#         }), 200

#     except Exception as e:
#         return jsonify({
#             'status': 'error',
#             'message': f'Server error: {str(e)}'
#         }), 500

# @app.route('/test', methods=['GET'])
# def test_with_sample_data():
#     """
#     Test endpoint using hardcoded sample data from your CSV
#     This simulates a real request without needing live camera input
#     """
#     try:
#         # Test with sample data
#         test_posture = "push_up"
#         test_landmarks = SAMPLE_TEST_DATA["landmarks"]
#         test_angles = SAMPLE_TEST_DATA["angles"]
        
#         feedback = {}
        
#         # Test landmarks comparison
#         landmarks_array = np.array(test_landmarks, dtype=float)
#         landmarks_feedback = COMPARISON_FUNCTIONS[test_posture]["landmarks"](landmarks_array)
#         feedback['landmarks'] = landmarks_feedback
        
#         # Test angles comparison
#         angles_array = np.array(test_angles, dtype=float)
#         angles_feedback = COMPARISON_FUNCTIONS[test_posture]["angles"](angles_array)
#         feedback['angles'] = angles_feedback
        
#         return jsonify({
#             'status': 'success',
#             'message': 'Test completed with sample data',
#             'posture': test_posture,
#             'feedback': feedback,
#             'test_data_used': {
#                 'landmarks_count': len(test_landmarks),
#                 'angles_count': len(test_angles)
#             }
#         }), 200
        
#     except Exception as e:
#         return jsonify({
#             'status': 'error',
#             'message': f'Test failed: {str(e)}'
#         }), 500

# if __name__ == '__main__':
#     print("Starting Pose Detection API...")
#     print("Available endpoints:")
#     print("  POST /pose - Main pose analysis endpoint")
#     print("  GET /test - Test with sample CSV data")
#     print("\nTest with sample data: curl http://localhost:5000/test")
    
#     app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd

from app.pose_logic import pushup_landmarks, pushup_angles

app = Flask(__name__)
CORS(app)

# Fixed test data - 99 landmarks (33 points × 3 coordinates) and 7 angles
SAMPLE_TEST_DATA = {
    "landmarks": [
        0,208,-0.14172298,-55.86816,-83.562836,0.77226424,-57.81948,-78.75882,1.233074,-57.699406
        ,-78.759254,1.7493129,-57.53372,-78.78233,-1.3052675,-57.72683,-79.61645,-2.2151628,-57.54617
        ,-79.630646,-3.1707785,-57.298553,-79.6179,2.756506,-55.85471,-50.886845,-4.7613044,-55.44263
        ,-54.861935,1.1918291,-53.003086,-73.335304,-1.6319023,-52.80554,-74.4462,9.726373,-40.375786
        ,-32.200623,-10.973151,-39.614494,-36.86989,26.051504,-39.137066,-31.208908,-27.616627,-37.245445
        ,-37.595387,38.416695,-47.548244,-57.050426,-41.644993,-45.20416,-64.12199,42.235176,-49.71335
        ,-63.664894,-45.83316,-47.217133,-71.61033,42.45105,-51.35462,-70.81914,-45.726147,-48.86608
        ,-78.51114,40.822147,-50.241287,-61.593933,-44.12522,-47.887577,-68.40401,5.9720097,-0.1093643
        ,0.26248583,-5.9720097,0.1093643,-0.26248583,11.75684,31.83908,-4.643736,-11.081656,32.370235
        ,-5.003058,14.448328,61.644,31.237148,-14.260321,62.669857,27.827307,13.20743,64.77579,32.967335
        ,-13.362365,66.36444,29.356094,16.494827,74.145805,1.6187178,-16.030586,74.7956,-1.0190959
    ],
    "angles": [0,208,90.98465,88.599945,38.76677,124.538,121.06854,116.595764,109.79157]
}

SAMPLE_TEST_DATA_2 = {
    "landmarks": [
        0,214,-0.09433113,-52.82708,-57.42315,0.5958145,-54.782166,-51.497883,1.2533442,-54.74263
        ,-51.483875,1.7506169,-54.66264,-51.489212,-1.3231053,-54.442043,-51.81584,-2.0938802,-54.147053
        ,-51.82886,-2.8000605,-53.831818,-51.82372,2.6459832,-52.775394,-25.84235,-3.6822598,-51.875507
        ,-26.83165,1.3602847,-49.7299,-48.547035,-1.1088375,-49.32755,-48.91358,8.897671,-39.977417
        ,-19.134594,-8.778292,-40.022503,-19.594484,17.166763,-56.413048,-35.11141,-18.693428,-55.127407
        ,-35.151936,12.452961,-73.303955,-49.73784,-14.891015,-74.892845,-45.190125,11.785952,-78.445435
        ,-57.70145,-13.212663,-80.25727,-54.24714,10.43839,-78.664055,-56.0308,-11.621859,-80.60166,-50.147144
        ,10.34366,-76.88346,-50.40773,-11.851206,-78.92865,-44.997295,5.7404513,-0.35265207,0.33231744
        ,-5.7404513,0.35264072,-0.33231744,14.389475,27.73498,-0.55136734,-10.437229,28.616667,0.030994494
        ,19.635443,51.882656,45.14488,-19.634829,51.199955,43.19759,18.795805,55.034233,48.25898
        ,-20.250454,54.455738,46.124863,21.455263,60.002815,19.192198,-21.453554,59.534573,18.457273
    ],
    "angles": [0,214,151.51472,156.4084,47.469006,119.39075,116.57121,141.04236,148.08401]
}

# Map posture names to comparison functions
COMPARISON_FUNCTIONS = {
    "push_up": {
        "landmarks": pushup_landmarks,
        "angles": pushup_angles
    }
}

def validate_input_data(landmarks, angles, posture):
    """
    Validate input data dimensions and format.
    
    Returns:
        tuple: (is_valid: bool, error_message: str, corrected_data: dict)
    """
    corrected_data = {"landmarks": landmarks, "angles": angles}
    
    # Validate landmarks
    if landmarks:
        landmarks_array = np.array(landmarks, dtype=float)
        expected_landmark_dims = 99  # 33 pose landmarks × 3 coordinates
        
        if len(landmarks_array) < expected_landmark_dims:
            # Pad with zeros
            padding_needed = expected_landmark_dims - len(landmarks_array)
            landmarks_array = np.concatenate([landmarks_array, np.zeros(padding_needed)])
            corrected_data["landmarks"] = landmarks_array.tolist()
            print(f"Padded landmarks from {len(landmarks)} to {expected_landmark_dims}")
        elif len(landmarks_array) > expected_landmark_dims:
            # Truncate
            landmarks_array = landmarks_array[:expected_landmark_dims]
            corrected_data["landmarks"] = landmarks_array.tolist()
            print(f"Truncated landmarks from {len(landmarks)} to {expected_landmark_dims}")
    
    # Validate angles
    if angles:
        angles_array = np.array(angles, dtype=float)
        expected_angle_dims = 7  # Standard joint angles for pose analysis
        
        if len(angles_array) < expected_angle_dims:
            # Pad with average angle (90 degrees)
            padding_needed = expected_angle_dims - len(angles_array)
            angles_array = np.concatenate([angles_array, np.full(padding_needed, 90.0)])
            corrected_data["angles"] = angles_array.tolist()
            print(f"Padded angles from {len(angles)} to {expected_angle_dims}")
        elif len(angles_array) > expected_angle_dims:
            # Truncate
            angles_array = angles_array[:expected_angle_dims]
            corrected_data["angles"] = angles_array.tolist()
            print(f"Truncated angles from {len(angles)} to {expected_angle_dims}")
    
    return True, "", corrected_data

@app.route('/pose', methods=['POST'])
def receive_pose():
    """
    Main endpoint for pose analysis
    Expected JSON format:
    {
        "posture": "push_up",
        "landmarks": [99 float values],
        "angles": [7 float values]
    }
    """
    try:
        data = request.get_json()
        posture = data.get('posture')
        landmarks = data.get('landmarks', [])
        angles = data.get('angles', [])

        if posture not in COMPARISON_FUNCTIONS:
            return jsonify({
                'status': 'error', 
                'message': f'Unknown posture: {posture}. Supported: {list(COMPARISON_FUNCTIONS.keys())}'
            }), 400

        # Validate and correct input data dimensions
        is_valid, error_msg, corrected_data = validate_input_data(landmarks, angles, posture)
        
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': error_msg
            }), 400

        feedback = {}
        
        # Process landmarks if provided
        if corrected_data["landmarks"]:
            try:
                landmarks_array = np.array(corrected_data["landmarks"], dtype=float)
                landmarks_feedback = COMPARISON_FUNCTIONS[posture]["landmarks"](landmarks_array)
                feedback['landmarks'] = landmarks_feedback
            except Exception as e:
                feedback['landmarks'] = {
                    "message": f"Error processing landmarks: {str(e)}",
                    "error": True
                }

        # Process angles if provided
        if corrected_data["angles"]:
            try:
                angles_array = np.array(corrected_data["angles"], dtype=float)
                angles_feedback = COMPARISON_FUNCTIONS[posture]["angles"](angles_array)
                feedback['angles'] = angles_feedback
            except Exception as e:
                feedback['angles'] = {
                    "message": f"Error processing angles: {str(e)}",
                    "error": True
                }

        return jsonify({
            'status': 'success',
            'posture': posture,
            'feedback': feedback,
            'data_corrections': {
                'landmarks_original_length': len(landmarks) if landmarks else 0,
                'landmarks_corrected_length': len(corrected_data["landmarks"]) if corrected_data["landmarks"] else 0,
                'angles_original_length': len(angles) if angles else 0,
                'angles_corrected_length': len(corrected_data["angles"]) if corrected_data["angles"] else 0
            }
        }), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/test', methods=['GET'])
def test_with_sample_data():
    """
    Test endpoint using hardcoded sample data from your CSV
    This simulates a real request without needing live camera input
    """
    try:
        # Test with sample data
        test_posture = "push_up"
        test_landmarks = SAMPLE_TEST_DATA["landmarks"]
        test_angles = SAMPLE_TEST_DATA["angles"]
        
        # Validate dimensions
        print(f"Test data - Landmarks: {len(test_landmarks)}, Angles: {len(test_angles)}")
        
        # Validate and correct input data
        is_valid, error_msg, corrected_data = validate_input_data(test_landmarks, test_angles, test_posture)
        
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': f'Test data validation failed: {error_msg}'
            }), 500
        
        feedback = {}
        
        # Test landmarks comparison
        try:
            landmarks_array = np.array(corrected_data["landmarks"], dtype=float)
            landmarks_feedback = COMPARISON_FUNCTIONS[test_posture]["landmarks"](landmarks_array)
            feedback['landmarks'] = landmarks_feedback
        except Exception as e:
            feedback['landmarks'] = {
                "message": f"Landmarks test failed: {str(e)}",
                "error": True
            }
        
        # Test angles comparison
        try:
            angles_array = np.array(corrected_data["angles"], dtype=float)
            angles_feedback = COMPARISON_FUNCTIONS[test_posture]["angles"](angles_array)
            feedback['angles'] = angles_feedback
        except Exception as e:
            feedback['angles'] = {
                "message": f"Angles test failed: {str(e)}",
                "error": True
            }
        
        return jsonify({
            'status': 'success',
            'message': 'Test completed with sample data',
            'posture': test_posture,
            'feedback': feedback,
            'test_data_used': {
                'landmarks_original_count': len(test_landmarks),
                'landmarks_final_count': len(corrected_data["landmarks"]),
                'angles_original_count': len(test_angles),
                'angles_final_count': len(corrected_data["angles"])
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Test failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("Starting Pose Detection API...")
    print("Available endpoints:")
    print("  POST /pose - Main pose analysis endpoint")
    print("  GET /test - Test with sample CSV data")
    print("\nTest commands:")
    print("  curl http://localhost:5000/test")
  
    
    app.run(host='0.0.0.0', port=5000, debug=True)