import numpy as np
import matplotlib.pyplot as plt
import os
from data_loader import pushup_landmarks, jumping_jack_landmarks, situp_landmarks, squat_landmarks, pullup_landmarks

all_lines = jumping_jack_landmarks()  # shape: (N, 99)
batch_size = 100
output_dir = "jumping_jack_pose"
os.makedirs(output_dir, exist_ok=True)

POSE_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 12), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 31), (28, 32),
    (15, 21), (16, 22),
    (11, 23), 
    (12, 24),
]

num_batches = len(all_lines) // batch_size + (1 if len(all_lines) % batch_size else 0)

for batch_idx in range(num_batches):
    start = batch_idx * batch_size
    end = min((batch_idx + 1) * batch_size, len(all_lines))
    lines = all_lines[start:end]

    fig, axes = plt.subplots(10, 10, figsize=(40, 40))
    for idx, (line, ax) in enumerate(zip(lines, axes.flatten())):
        landmarks = np.array(line).reshape((33, 3))
        ax.scatter(landmarks[:, 0], -landmarks[:, 1], c='r', s=10)
        hand_idx = [15, 17, 19, 21, 16, 18, 20, 22]
        ax.scatter(landmarks[hand_idx, 0], -landmarks[hand_idx, 1], c='b', s=20, label='Hands')
        leg_idx = [23, 25, 27, 29, 31, 24, 26, 28, 30, 32]
        ax.scatter(landmarks[leg_idx, 0], -landmarks[leg_idx, 1], c='g', s=20, label='Legs')
        for a, b in POSE_CONNECTIONS:
            ax.plot([landmarks[a, 0], landmarks[b, 0]], [-landmarks[a, 1], -landmarks[b, 1]], 'b')
        ax.set_title(f"Pose {start + idx + 1}")
        ax.axis('equal')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"poses_{start}_{end-1}.png"))
    plt.close(fig)