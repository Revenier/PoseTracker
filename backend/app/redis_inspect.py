import redis
import pickle
import hashlib
import binascii

def get_redis_size(r):
    total_bytes = 0
    keys = r.keys()
    print(f"🔎 Redis contains {len(keys)} keys:\n")

    for key in keys:
        try:
            value = r.get(key)
            size = len(value)
            total_bytes += size

            # Coba load, dan ambil function name jika mungkin
            try:
                data = pickle.loads(value)

                if isinstance(data, dict) and 'function' in data:
                    func_name = data['function']
                elif isinstance(data, tuple) and len(data) > 0 and isinstance(data[0], str):
                    func_name = data[0]
                else:
                    func_name = "<unknown>"
            except Exception:
                func_name = "<unreadable>"

            # Cetak key dan size
            print(f"Key: {binascii.hexlify(key).decode()[:20]}... | Size: {size} bytes | Function: {func_name}")
        except Exception as e:
            print(f"❌ Error reading key {key}: {e}")

    kb = total_bytes / 1024
    print(f"\n📦 Total Redis data size: {kb:.2f} KB ({total_bytes} bytes)")


if __name__ == "__main__":
    r = redis.Redis(host='localhost', port=6379, db=0)
    get_redis_size(r)
