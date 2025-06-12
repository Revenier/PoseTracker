from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

from app.redis_client import get_pose_result
from app.pose_logic import (
    pushup_landmarks, pushup_angles,
    situp_landmarks, situp_angles,
    squat_landmarks, squat_angles,
    pullup_landmarks, pullup_angles,
    jumping_jack_landmarks, jumping_jack_angles,
)

app = Flask(__name__)
CORS(app)

# {
#   "posture": "push_up",
#   "landmarks": [...],   // 33*3 = 99 floats
#   "angles": [...]       // list of angles
# }

# Map posture names to reference data
LANDMARKS_MAP = {
    "push_up": pushup_landmarks,
    "sit_up": situp_landmarks,
    "squat": squat_landmarks,
    "pull_up": pullup_landmarks,
    "jumping_jack": jumping_jack_landmarks,
}
ANGLES_MAP = {
    "push_up": pushup_angles,
    "sit_up": situp_angles,
    "squat": squat_angles,
    "pull_up": pullup_angles,
    "jumping_jack": jumping_jack_angles,
}

@app.route('/pose', methods=['POST'])
def receive_pose():
    data = request.get_json()
    posture = data.get('posture')
    landmarks = data.get('landmarks', [])
    angles = data.get('angles', [])

    if posture not in LANDMARKS_MAP or posture not in ANGLES_MAP:
        return jsonify({'status': 'error', 'message': 'Unknown posture'}), 400

    feedback = {}

    return jsonify({
        'status': 'success',
        'feedback': feedback
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)