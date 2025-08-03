# Script ini memuat data pickle hasil training (landmark dan angle) ke dalam Redis untuk mempercepat akses saat inferensi

from app.data_loader import pushup_angles, pushup_landmarks
from app.logic import angle_logic, landmark_logic
from app.redis_client import get_or_cache_result
import numpy as np

POSTURE = 'push_up'
PRINT_LIMIT = 10

print(f"\nMulai preload posture: {POSTURE}\n")

angles_array = pushup_angles()
landmarks_array = pushup_landmarks()

print(f"→ Total: {len(angles_array)} angle frames")
print(f"→ Total: {len(landmarks_array)} landmark frames\n")

print("Preloading LANDMARK frames:\n")
for i, frame in enumerate(landmarks_array[:PRINT_LIMIT]):
    frame = np.array(frame, dtype=np.float64)
    result = get_or_cache_result((POSTURE, frame), landmark_logic)
    print(f"[LANDMARK {i+1}] Result: {result}")

print("\nPreloading ANGLE frames (dihitung dari landmark):\n")
for i, frame in enumerate(landmarks_array[:PRINT_LIMIT]):
    frame = np.array(frame, dtype=np.float64)
    result = get_or_cache_result((POSTURE, frame), angle_logic)
    print(f"[ANGLE {i+1}] Result: {result}")

print("\nSelesai simpan 10 pertama landmark & angle (via landmark) ke Redis.\n")

