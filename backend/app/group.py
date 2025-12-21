from collections import Counter

bodyPartMap = {
    # Head
    0: "head", 1: "head", 2: "head", 3: "head", 4: "head", 5: "head", 6: "head", 7: "head", 8: "head", 9: "head", 10: "head",
    # Left Arm
    11: "left arm", 13: "left arm", 15: "left arm", 17: "left arm", 19: "left arm", 21: "left arm",
    # Right Arm
    12: "right arm", 14: "right arm", 16: "right arm", 18: "right arm", 20: "right arm", 22: "right arm",
    # Left Leg
    23: "left leg", 25: "left leg", 27: "left leg", 29: "left leg", 31: "left leg",
    # Right Leg
    24: "right leg", 26: "right leg", 28: "right leg", 30: "right leg", 32: "right leg"
}

ANGLE_NAME_MAP = {
    (14, 12, 24): "right_shoulder_angle",
    (13, 11, 23): "left_shoulder_angle",
    (26, 24, 25): "right_knee_angle",
    (24, 26, 28): "right_hip_angle",
    (23, 25, 27): "left_hip_angle",
    (16, 14, 12): "right_wrist_angle",
    (15, 13, 11): "left_wrist_angle",
}

def group_landmark_issue(results, score):
    if not isinstance(results, list) or len(results) == 0:
        return "No feedback available"

    total = len(results)
    correct_count = sum(1 for r in results if r.get('landmarks', {}).get('correct', False))

    feedbacks = []
    for r in results:
        if not isinstance(r, dict):
            continue
        lm = r.get('landmarks', {})
        fb = lm.get('issues')
        if fb and not lm.get('correct', False):
            feedbacks.append(str(fb))

    top = Counter(feedbacks).most_common(1)
    top_feedback = top[0][0] if top else ""

    if correct_count == total:
        return "Perfect form!"
    elif score >= 80:
        return f"Good. {top_feedback}"
    else:
        return f"Bad. {top_feedback}"
    

def group_landmark_feedback(results):
    if not isinstance(results, list) or len(results) == 0:
        return "No feedback available"

    total = len(results)
    correct_count = sum(1 for r in results if r.get('landmarks', {}).get('correct', False))

    feedbacks = []
    for r in results:
        if not isinstance(r, dict):
            continue
        lm = r.get('landmarks', {})
        fb = lm.get('feedback')
        if fb and not lm.get('correct', False):
            feedbacks.append(str(fb))

    top = Counter(feedbacks).most_common(1)
    top_feedback = top[0][0] if top else ""

    if correct_count == total:
        return "No Feedback!"
    else:
        return f"{top_feedback}"