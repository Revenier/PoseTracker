import os
import pickle

base_dir = os.path.dirname(os.path.abspath(__file__))
pickles_dir = os.path.join(base_dir, 'AlignedPickles2')

def load_pickle(filename):
    with open(os.path.join(pickles_dir, filename), 'rb') as f:
        return pickle.load(f)

def pushup_landmarks():
    return load_pickle('push_up_landmarks_aligned.pkl')

def pushup_angles():
    return load_pickle('push_up_angles.pkl')

def situp_landmarks():
    return load_pickle('situp_landmarks_aligned.pkl')

def situp_angles():
    return load_pickle('situp_angles.pkl')

def squat_landmarks():
    return load_pickle('squat_landmarks_aligned.pkl')

def squat_angles():
    return load_pickle('squat_angles.pkl')

def pullup_landmarks():
    return load_pickle('pull_up_landmarks_aligned.pkl')

def pullup_angles():
    return load_pickle('pull_up_angles.pkl')

def jumping_jack_landmarks():
    return load_pickle('jumping_jack_landmarks_aligned.pkl')

def jumping_jack_angles():
    return load_pickle('jumping_jack_angles.pkl')

posture_map = {
    'push_up': {
        'landmarks': pushup_landmarks,
        'angles': pushup_angles,
    },
    'sit_up': {
        'landmarks': situp_landmarks,
        'angles': situp_angles,
    },
    'squat': {
        'landmarks': squat_landmarks,
        'angles': squat_angles,
    },
    'pull_up': {
        'landmarks': pullup_landmarks,
        'angles': pullup_angles,
    },
    'jumping_jack': {
        'landmarks': jumping_jack_landmarks,
        'angles': jumping_jack_angles,
    }
}