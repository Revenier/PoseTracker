# Script ini digunakan untuk mengambil data mentah dari Redis dan mengubahnya menjadi format JSON yang bisa dibaca manusia.
# Data yang ditampilin itu bentuknya dari data yang akan di post dari postman

import redis
import pickle
import json
from dotenv import load_dotenv, find_dotenv
import os

""" Versi Lokal / Docker"""

"""
Cara dapetin keynya:
1. buka terminal dan masukin
    docker exec -it fitpipe-redis redis-cli
2. masukin command
    KEYS *
3. nanti muncul semua key yang ada di redis docker
4. Yang bisa dipake itu yang key landmark 
5. Ketika Copy harus ada push_up:landmark:

"""

r = redis.Redis(host='localhost', port=6379, db=0)

key = "push_up:landmark:372c305671ad2b8cef0a8cce805d626db469211e2dc52e4822061ebacf160f57"
data = r.get(key)

if data:
    try:
        obj = pickle.loads(data)
        print(f"\nType of object: {type(obj)}")

        # Convert numpy array ke list (kalau ada mediapipe)
        if isinstance(obj, dict) and 'mediapipe' in obj:
            try:
                arr = obj['mediapipe'].tolist()
                # Potong per 10 elemen, simpan horizontal
                grouped = [arr[i:i+10] for i in range(0, len(arr), 10)]
                obj['mediapipe'] = [" ".join(map(str, chunk)) for chunk in grouped]
            except:
                pass

            print("JSON Representation:\n")
            print(json.dumps(obj, indent=4))
    except Exception as e:
        print("Gagal decode pickle:", e)
else:
    print("Key tidak ditemukan.")


""" Versi Cloud """

# Script ini digunakan untuk mengambil data mentah dari Redis Cloud

"""
Cara dapetin keynya:
1. Buka RedisInsight
2. Masuk ke database Fitpipe Cloud
3. Klik salah satu key di database
4. Copy judul landmark (yang sebelah string)(harus ada push_up:landmark:)
"""

load_dotenv(find_dotenv())

# Redis Cloud untuk landmark
# r = redis.Redis(
#     host=os.getenv("REDIS_HOST_1"),
#     port=int(os.getenv("REDIS_PORT_1")),
#     username=os.getenv("REDIS_USER_1"),
#     password=os.getenv("REDIS_PASS_1"),
#     decode_responses=False  # penting untuk simpan/ambil data pickle
# )

# # Key yang mau dicek
# key = "push_up:landmark:1ee5f3312ce6f281857cc35088371014b393a88ff64e900ac714075b7323aa8f"

# data = r.get(key)

# if data:
#     try:
#         obj = pickle.loads(data)
#         print(f"\nType of object: {type(obj)}")

#         if isinstance(obj, dict) and 'mediapipe' in obj:
#             try:
#                 arr = obj['mediapipe'].tolist()
#                 # potong per 10 elemen
#                 grouped = [arr[i:i+10] for i in range(0, len(arr), 10)]
#                 # jadikan string horizontal
#                 obj['mediapipe'] = [" ".join(map(str, chunk)) for chunk in grouped]
#             except:
#                 pass

#             print("JSON Representation:\n")
#             print(json.dumps(obj, indent=4))
#     except Exception as e:
#         print("Gagal decode pickle:", e)
# else:
#     print("Key tidak ditemukan.")


