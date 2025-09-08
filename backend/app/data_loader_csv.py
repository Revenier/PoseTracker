import pandas as pd
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve()
DATA_PER_POSE = HERE.parent / "data_per_pose" 
def _read_csv(filename):
    path = DATA_PER_POSE / filename
    print(f"Trying to read: {path}")  
    df = pd.read_csv(path, header=0)
    return df.to_numpy(dtype=float)

def pushup_landmarks():       return _read_csv("push_up_landmarks_raw.csv")
def pushup_angles():          return _read_csv("push_up_angles_raw.csv")

def situp_landmarks():        return _read_csv("sit_up_landmarks_raw.csv")
def situp_angles():           return _read_csv("sit_up_angles_raw.csv")

def squat_landmarks():        return _read_csv("squat_landmarks_raw.csv")
def squat_angles():           return _read_csv("squat_angles_raw.csv")

def pullup_landmarks():       return _read_csv("pull_up_landmarks_raw.csv")
def pullup_angles():          return _read_csv("pull_up_angles_raw.csv")

def jumping_jack_landmarks(): return _read_csv("jumping_jack_landmarks_raw.csv")
def jumping_jack_angles():    return _read_csv("jumping_jack_angles_raw.csv")

posture_map = {
    "push_up":      {"landmarks": pushup_landmarks,      "angles": pushup_angles},
    "sit_up":       {"landmarks": situp_landmarks,       "angles": situp_angles},
    "squat":        {"landmarks": squat_landmarks,       "angles": squat_angles},
    "pull_up":      {"landmarks": pullup_landmarks,      "angles": pullup_angles},
    "jumping_jack": {"landmarks": jumping_jack_landmarks,"angles": jumping_jack_angles},
}
