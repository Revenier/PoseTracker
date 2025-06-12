import redis
import pickle
import hashlib

# Initialize Redis connection
r = redis.Redis(host='localhost', port=6379, db=0)

def get_pose_result(input_data, compare_func, *args):
    """
    Checks Redis for a cached result for the given input_data.
    If not found, runs compare_func(input_data, *args), caches, and returns the result.
    """
    # Create a unique key from the input data
    input_bytes = pickle.dumps(input_data)
    input_key = hashlib.sha256(input_bytes).hexdigest()

    cached = r.get(input_key)
    if cached:
        return pickle.loads(cached)
    # Run the comparison function
    result = compare_func(input_data, *args)
    r.set(input_key, pickle.dumps(result))
    return result