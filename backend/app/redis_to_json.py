#Script ini digunakan untuk mengambil data mentah dari Redis dan mengubahnya menjadi format JSON yang bisa dibaca manusia.

import redis
import pickle
import json

r = redis.Redis(host='localhost', port=6379, db=0)

key = "push_up:landmark:37a982075457d01d7404320e67a62d307565f43b50c8884c27fd965e61bf9385"
data = r.get(key)

if data:
    try:
        obj = pickle.loads(data)
        print(f"\nType of object: {type(obj)}")
        # print(f"Preview:\n{obj}\n")


        # Convert numpy array to list (if exists)
        if isinstance(obj, dict) and 'mediapipe' in obj:
            try:
                obj['mediapipe'] = obj['mediapipe'].tolist()
            except:
                pass

            print("JSON Representation:\n")
            print(json.dumps(obj, indent=4))
    except Exception as e:
        print("Gagal decode pickle:", e)
else:
    print("Key tidak ditemukan.")
