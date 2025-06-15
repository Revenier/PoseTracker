from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd

from app.pose_logic import pushup_landmarks, pushup_angles

app = Flask(__name__)
CORS(app)

# Test data from your CSV samples (hardcoded for testing)
SAMPLE_TEST_DATA = {
    "landmarks": [
        -0.14172298, -55.86816, -83.562836, 0.77226424, -57.81948, -78.75882, 
        1.233074, -57.699406, -78.759254, 1.7493129, -57.53372, -78.78233, 
        -1.3052675, -57.72683, -79.61645, -2.2151628, -57.54617, -79.630646, 
        -3.1707785, -57.298553, -79.6179, 2.756506, -55.85471, -50.886845, 
        -4.7613044, -55.44263, -54.861935, 1.1918291, -53.003086, -73.335304, 
        -1.6319023, -52.80554, -74.4462, 9.726373, -40.375786, -32.200623, 
        -10.973151, -39.614494, -36.86989, 26.051504, -39.137066, -31.208908, 
        -27.616627, -37.245445, -37.595387, 38.416695, -47.548244, -57.050426, 
        -41.644993, -45.20416, -64.12199, 42.235176, -49.71335, -63.664894, 
        -45.83316, -47.217133, -71.61033, 42.45105, -51.35462, -70.81914, 
        -45.726147, -48.86608, -78.51114, 40.822147, -50.241287, -61.593933, 
        -44.12522, -47.887577, -68.40401, 5.9720097, -0.1093643, 0.26248583, 
        -5.9720097, 0.1093643, -0.26248583, 11.75684, 31.83908, -4.643736, 
        -11.081656, 32.370235, -5.003058, 14.448328, 61.644, 31.237148, 
        -14.260321, 62.669857, 27.827307, 13.20743, 64.77579, 32.967335, 
        -13.362365, 66.36444, 29.356094, 16.494827, 74.145805, 1.6187178, 
        -16.030586, 74.7956, -1.0190959
    ],
    "angles": [90.98465, 88.599945, 38.76677, 124.538, 121.06854, 116.595764, 109.79157]
}

# Map posture names to comparison functions
COMPARISON_FUNCTIONS = {
    "push_up": {
        "landmarks": pushup_landmarks,
        "angles": pushup_angles
    }
}

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

        feedback = {}
        
        # Process landmarks if provided
        if landmarks:
            try:
                landmarks_array = np.array(landmarks, dtype=float)
                if len(landmarks_array) != 99:
                    return jsonify({
                        'status': 'error',
                        'message': f'Expected 99 landmark values, got {len(landmarks_array)}'
                    }), 400
                
                landmarks_feedback = COMPARISON_FUNCTIONS[posture]["landmarks"](landmarks_array)
                feedback['landmarks'] = landmarks_feedback
            except Exception as e:
                feedback['landmarks'] = f"Error processing landmarks: {str(e)}"

        # Process angles if provided
        if angles:
            try:
                angles_array = np.array(angles, dtype=float)
                angles_feedback = COMPARISON_FUNCTIONS[posture]["angles"](angles_array)
                feedback['angles'] = angles_feedback
            except Exception as e:
                feedback['angles'] = f"Error processing angles: {str(e)}"

        return jsonify({
            'status': 'success',
            'posture': posture,
            'feedback': feedback
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
        
        feedback = {}
        
        # Test landmarks comparison
        landmarks_array = np.array(test_landmarks, dtype=float)
        landmarks_feedback = COMPARISON_FUNCTIONS[test_posture]["landmarks"](landmarks_array)
        feedback['landmarks'] = landmarks_feedback
        
        # Test angles comparison
        angles_array = np.array(test_angles, dtype=float)
        angles_feedback = COMPARISON_FUNCTIONS[test_posture]["angles"](angles_array)
        feedback['angles'] = angles_feedback
        
        return jsonify({
            'status': 'success',
            'message': 'Test completed with sample data',
            'posture': test_posture,
            'feedback': feedback,
            'test_data_used': {
                'landmarks_count': len(test_landmarks),
                'angles_count': len(test_angles)
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
    print("\nTest with sample data: curl http://localhost:5000/test")
    
    app.run(host='0.0.0.0', port=5000, debug=True)