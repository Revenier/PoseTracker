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

def group_feedback(feedback_list):
    # Collect all wrong indices from the batch
    all_wrong_indices = []
    for fb in feedback_list:
        if isinstance(fb, dict) and 'wrong_indices' in fb:
            all_wrong_indices.extend(fb['wrong_indices'])

    if not all_wrong_indices:
        return "Correct posture!"

    # Count which body parts are most often wrong
    body_parts = [bodyPartMap.get(idx, "unknown") for idx in all_wrong_indices]
    part_counts = Counter(body_parts)
    # Get the 2 most common mistakes
    most_common = part_counts.most_common(2)

    # Build feedback message
    feedback_msgs = [f"Check your {part} (mistakes: {count})" for part, count in most_common]
    return " | ".join(feedback_msgs)

# Example usage:
# feedback_list = [
#     {"wrong_indices": [13, 25]},  # left arm, left leg
#     {"wrong_indices": [25]},      # left leg
#     {"wrong_indices": []},        # correct
# ]
# print(group_feedback(feedback_list))