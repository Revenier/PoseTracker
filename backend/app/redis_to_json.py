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

REDIS_HOST = "localhost"      # atau "redis" kalau di docker
REDIS_PORT = 6379
REDIS_DB   = 0

# ganti key di sini
key = "push_up:angle:260a28e7063e9659fc935cdc5a0c60834234211c0b87d0b770533a9709292885"

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
data = r.get(key)

if not data:
    print(f"❌ Key '{key}' tidak ditemukan di Redis.")
else:
    try:
        obj = pickle.loads(data)
        print(f"✅ Key ditemukan: {key}")
        print(f"📦 Tipe objek: {type(obj)}")

        # kalau numpy array, ubah ke list biar bisa dibaca
        if hasattr(obj, "tolist"):
            obj = obj.tolist()

        # pastikan flat list
        if isinstance(obj, list) and isinstance(obj[0], (int, float)):
            flat = obj
        elif isinstance(obj, list) and isinstance(obj[0], list):
            # misal 2D array, flatten aja
            flat = [x for row in obj for x in row]
        else:
            print("⚠️ Format data tidak terduga, menampilkan JSON mentah:")
            print(json.dumps(obj, indent=2))
            exit()

        # tampilkan 10 angka per baris
        print("\n📊 Data isi:")
        for i in range(0, len(flat), 10):
            chunk = flat[i:i+10]
            print(" ".join(f"{v:.5f}" for v in chunk))

        print(f"\n🔢 Total elemen: {len(flat)}")
    except Exception as e:
        print("⚠️ Gagal decode pickle:", e)