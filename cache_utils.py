import os
import json
import threading

CACHE_FILE = "prompt_response_cache.json"
_LOCK = threading.Lock()

def _load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def _save_cache(cache_dict):
    with _LOCK:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_dict, f, indent=2, ensure_ascii=False)

def get_cached_response(prompt: str):
    cache = _load_cache()
    return cache.get(prompt.strip().lower())  # Normalize keys

def set_cache_response(prompt: str, response: str):
    cache = _load_cache()
    cache[prompt.strip().lower()] = response
    _save_cache(cache)

def get_last_n_queries(n=5):
    """Return last n queries from cache in insertion order (newest last)."""
    cache = _load_cache()
    # JSON dicts are insertion-ordered in Python 3.7+
    keys = list(cache.keys())
    return keys[-n:] if len(keys) >= n else keys
